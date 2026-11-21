/**
 * Custom code-mode executor example for @mcp-use/client.
 *
 * Shows how to plug your own runtime into code mode by subclassing the
 * exported `BaseCodeExecutor`, with no changes to @mcp-use/client itself.
 * Each `executeCode()` call runs the agent's code in a separate Node.js child
 * process. Tool calls made inside the child are sent to the host over IPC,
 * executed against the real MCP session, and the result is sent back. A
 * remote sandbox (E2B, or any vendor SDK) uses the same pattern with a
 * different transport.
 *
 * A child process is NOT a security boundary: it has the same filesystem and
 * network access as the host. Use a real sandbox for untrusted code.
 *
 * The example connects to the official filesystem MCP server via `npx`, so it
 * needs network access on first run but no API keys or paid services.
 *
 * Run (from packages/client):
 *   pnpm build
 *   pnpm dlx tsx examples/node/code-mode-custom-executor.ts
 */

import { fork, type ChildProcess } from "node:child_process";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import {
  BaseCodeExecutor,
  MCPClient,
  type CodeExecutorFunction,
} from "@mcp-use/client";

// `ExecutionResult` is not re-exported from the package root, but it is the
// resolved return type of the exported `CodeExecutorFunction`.
type ExecutionResult = Awaited<ReturnType<CodeExecutorFunction>>;

/** Messages the child process sends to the host. */
type ChildMessage =
  | { type: "log"; line: string }
  | {
      type: "tool_call";
      id: number;
      server: string;
      tool: string;
      args: Record<string, unknown>;
    }
  | { type: "search_tools"; id: number; query: string; detailLevel: string }
  | { type: "done"; result: unknown }
  | { type: "failed"; error: string };

/**
 * Code that runs inside the child before the agent's code. It defines one
 * global object per MCP server whose methods forward calls to the host, plus a
 * `search_tools` helper, and captures `console` output as logs.
 */
function buildChildScript(
  catalog: Record<string, string[]>,
  code: string
): string {
  return `
let nextId = 0;
const pending = new Map();
process.on("message", (msg) => {
  const entry = pending.get(msg.id);
  if (!entry) return;
  pending.delete(msg.id);
  if (msg.error) entry.reject(new Error(msg.error));
  else entry.resolve(msg.data);
});
const request = (payload) =>
  new Promise((resolve, reject) => {
    const id = nextId++;
    pending.set(id, { resolve, reject });
    process.send({ ...payload, id });
  });

const format = (a) => (typeof a === "string" ? a : JSON.stringify(a));
const log = (...args) => process.send({ type: "log", line: args.map(format).join(" ") });
console.log = console.info = console.debug = log;
console.warn = (...args) => log("[WARN]", ...args);
console.error = (...args) => log("[ERROR]", ...args);

globalThis.search_tools = (query = "", detailLevel = "full") =>
  request({ type: "search_tools", query, detailLevel });

const catalog = ${JSON.stringify(catalog)};
for (const [server, tools] of Object.entries(catalog)) {
  const ns = {};
  for (const tool of tools) {
    ns[tool] = (args = {}) => request({ type: "tool_call", server, tool, args });
  }
  globalThis[server] = ns;
  globalThis[server.replace(/[^a-zA-Z0-9_]/g, "_")] = ns;
}

(async () => {
${code}
})().then(
  (result) => process.send({ type: "done", result }, () => process.exit(0)),
  (e) => process.send({ type: "failed", error: e?.message ?? String(e) }, () => process.exit(0))
);
`;
}

class ChildProcessCodeExecutor extends BaseCodeExecutor {
  private workDir: string | null = null;
  private running = new Set<ChildProcess>();

  constructor(client?: MCPClient) {
    super(client as MCPClient);
  }

  /** Bind the executor to the client it serves (see `main()` below). */
  attach(client: MCPClient): this {
    this.client = client;
    return this;
  }

  /** Lazily create resources on first use, like a remote sandbox would. */
  private async getWorkDir(): Promise<string> {
    this.workDir ??= await mkdtemp(join(tmpdir(), "mcp-code-mode-"));
    return this.workDir;
  }

  async execute(code: string, timeout = 30_000): Promise<ExecutionResult> {
    const start = Date.now();
    const logs: string[] = [];
    let result: unknown = null;
    let error: string | null = null;

    try {
      // Connect any configured servers that are not connected yet.
      await this.ensureServersConnected();

      // Tool names per server, from the client's active sessions.
      const catalog: Record<string, string[]> = {};
      for (const { serverName, tools } of this.getToolNamespaces()) {
        catalog[serverName] = tools.map((t) => t.name);
      }

      const file = join(await this.getWorkDir(), `exec_${Date.now()}.mjs`);
      await writeFile(file, buildChildScript(catalog, code));

      result = await this.runChild(file, timeout, logs);
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
    }

    return {
      result,
      logs,
      error,
      execution_time: (Date.now() - start) / 1000,
    };
  }

  private runChild(
    file: string,
    timeout: number,
    logs: string[]
  ): Promise<unknown> {
    return new Promise((resolvePromise, reject) => {
      const child = fork(file, [], {
        stdio: ["ignore", "inherit", "inherit", "ipc"],
      });
      this.running.add(child);

      const timer = setTimeout(() => {
        child.kill("SIGKILL");
        reject(new Error("Script execution timed out"));
      }, timeout);

      const finish = () => {
        clearTimeout(timer);
        this.running.delete(child);
      };

      child.on("message", async (msg: ChildMessage) => {
        switch (msg.type) {
          case "log":
            logs.push(msg.line);
            return;
          case "done":
            finish();
            resolvePromise(msg.result);
            return;
          case "failed":
            finish();
            reject(new Error(msg.error));
            return;
          case "search_tools": {
            const search = this.createSearchToolsFunction();
            const data = await search(
              msg.query,
              msg.detailLevel as "names" | "descriptions" | "full"
            );
            child.send({ id: msg.id, data });
            return;
          }
          case "tool_call":
            child.send({ id: msg.id, ...(await this.callTool(msg)) });
            return;
        }
      });

      child.on("exit", (exitCode, signal) => {
        if (!this.running.has(child)) return;
        finish();
        reject(
          new Error(`Child exited early (code ${exitCode}, signal ${signal})`)
        );
      });
    });
  }

  /** Run a tool call from the child against the real MCP session. */
  private async callTool(msg: {
    server: string;
    tool: string;
    args: Record<string, unknown>;
  }): Promise<{ data?: unknown; error?: string }> {
    try {
      const session = this.client.getAllActiveSessions()[msg.server];
      if (!session) throw new Error(`Server ${msg.server} not found`);
      const toolResult = await session.callTool(msg.tool, msg.args);
      if (toolResult.isError) {
        const first = toolResult.content?.[0];
        throw new Error(
          first?.type === "text" ? first.text : "Tool call failed"
        );
      }
      // Same unwrapping as the built-in executors: parse the first text item.
      const first = toolResult.content?.[0];
      if (first?.type === "text") {
        try {
          return { data: JSON.parse(first.text) };
        } catch {
          return { data: first.text };
        }
      }
      return { data: first ?? toolResult };
    } catch (e) {
      return { error: e instanceof Error ? e.message : String(e) };
    }
  }

  /** Called by `client.close()`. Release everything this executor created. */
  async cleanup(): Promise<void> {
    for (const child of this.running) child.kill("SIGKILL");
    this.running.clear();
    if (this.workDir) {
      await rm(this.workDir, { recursive: true, force: true });
      this.workDir = null;
    }
  }
}

async function main(): Promise<void> {
  const directory = resolve("./tmp-code-mode-example");
  await mkdir(directory, { recursive: true });

  // `BaseCodeExecutor` keeps a reference to the client whose sessions it uses,
  // and `MCPClient` takes the executor in its constructor options. Create the
  // executor first, then attach the client it belongs to. The executor is not
  // used until the first `executeCode()` call.
  const executor = new ChildProcessCodeExecutor();
  const client = new MCPClient(
    {
      mcpServers: {
        fs: {
          command: "npx",
          args: ["-y", "@modelcontextprotocol/server-filesystem", directory],
        },
      },
    },
    { codeMode: { enabled: true, executor } }
  );
  executor.attach(client);

  try {
    const run = await client.executeCode(`
      await fs.write_file({ path: ${JSON.stringify(join(directory, "hello.txt"))}, content: "hi from a child process" });
      const found = await search_tools("directory", "names");
      console.log("directory tools:", found.results.map((t) => t.name).join(", "));
      console.log("pid differs from host:", process.pid !== ${process.pid});
      return await fs.list_directory({ path: ${JSON.stringify(directory)} });
    `);
    console.log("result:", run.result);
    console.log("logs:", run.logs);
    console.log("error:", run.error);

    const slow = await client.executeCode("await new Promise(() => {});", 500);
    console.log("timeout error:", slow.error);

    const failing = await client.executeCode(
      `return await fs.read_text_file({ path: "/definitely/not/allowed" });`
    );
    console.log("tool error:", failing.error);
  } finally {
    await client.close();
    await rm(directory, { recursive: true, force: true });
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
