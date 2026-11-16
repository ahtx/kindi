// Copyright (c) 2026 Amir Husain. SPDX-License-Identifier: AGPL-3.0-only. See LICENSE and COMMERCIAL-LICENSE.md.
// The same native pi UI as kindi-teaser; Bridge observes it and queues scheduled turns.
import { spawn } from "node:child_process";
import { StringDecoder } from "node:string_decoder";

export default function (pi: any) {
  let child: any, ctx: any, timer: any, buffer = "", activeJob: string | undefined;
  let offInput: any;
  const pending = new Map<string, (text: string) => void>();
  const send = (record: any) => {
    if (child?.stdin?.writable) child.stdin.write(JSON.stringify(record) + "\n");
  };
  const ready = () => {
    if (ctx && !activeJob) send({ type: "ready", idle: ctx.isIdle() && !ctx.hasPendingMessages(), editing: !!ctx.ui.getEditorText() });
  };
  const cleanup = async () => {
    const closing = child;
    clearInterval(timer); offInput?.(); offInput = undefined;
    for (const resolve of pending.values()) resolve(""); pending.clear();
    if (child) { send({ type: "shutdown" }); child.stdin.end(); child = undefined; }
    if (closing && closing.exitCode === null) await new Promise<void>((resolve) => {
      const timeout = setTimeout(() => { closing.kill(); resolve(); }, 2000);
      closing.once("exit", () => { clearTimeout(timeout); resolve(); });
    });
  };
  pi.on("session_start", async (_: any, context: any) => {
    await cleanup();
    ctx = context; buffer = ""; activeJob = undefined;
    child = spawn(process.env.BRIDGE_PYTHON!, ["-m", "kindi_bridge", "native-adapter", process.env.BRIDGE_NATIVE_AGENT!], { stdio: ["pipe", "pipe", "pipe"], env: process.env });
    ctx.ui.setHeader((_tui: any, theme: any) => ({
      invalidate() {},
      render(width: number) {
        const label = "KINDI // " + (process.env.BRIDGE_NATIVE_NAME || "default");
        return [theme.fg("accent", label.slice(0, width)), theme.fg("dim", "Type a mission or prompt here. Alt-M opens the menu.".slice(0, width)), ""];
      },
    }));
    ctx.ui.setStatus("bridge", "Agents: switch conversations or assign a mission from the menu.");
    child.on("error", (e: any) => ctx.ui.notify("Bridge adapter: " + e.message, "error"));
    child.stderr.on("data", (data: any) => ctx.ui.notify("Bridge: " + data.toString().slice(-300), "warning"));
    const decoder = new StringDecoder("utf8");
    child.stdout.on("data", (data: any) => {
      buffer += decoder.write(data);
      let end: number;
      while ((end = buffer.indexOf("\n")) >= 0) {
        const line = buffer.slice(0, end); buffer = buffer.slice(end + 1);
        let record: any;
        try { record = JSON.parse(line); } catch { continue; }
        if (record.type === "retrieval") {
          pending.get(record.id)?.(record.text); pending.delete(record.id);
        } else if (record.type === "prompt") {
          if (!ctx.isIdle() || ctx.hasPendingMessages() || ctx.ui.getEditorText() || activeJob) {
            send({ type: "deferred", id: record.id }); continue;
          }
          activeJob = record.id;
          try {
            // Public ExtensionAPI returns void; some embedded implementations return a Promise.
            Promise.resolve(pi.sendUserMessage(record.text, { deliverAs: "followUp", expandPromptTemplates: false })).catch((e: any) => {
              send({ type: "delivery_error", id: record.id, error: String(e) }); activeJob = undefined;
            });
          } catch (e) { send({ type: "delivery_error", id: record.id, error: String(e) }); activeJob = undefined; }
        }
      }
    });
    const sessionChild = child;
    child.on("exit", () => { if (child === sessionChild) ctx.ui.setStatus("bridge", "BRIDGE · scheduler adapter offline"); });
    offInput = ctx.ui.onTerminalInput(() => { setTimeout(ready, 0); });
    timer = setInterval(ready, 500); ready();
  });
  pi.on("before_agent_start", async (event: any) => {
    if (activeJob || event.messageType !== "user") return;
    send({ type: "manual_prompt", text: event.prompt });
    const id = "rag-" + Date.now();
    const text = await new Promise<string>((resolve) => {
      const timeout = setTimeout(() => { pending.delete(id); resolve(""); }, 15000);
      pending.set(id, (value) => { clearTimeout(timeout); resolve(value); });
      send({ type: "retrieve", id, query: event.prompt });
    });
    if (text) return { message: { customType: "bridge-reference", content: text, display: false } };
  });
  for (const type of ["agent_start", "message_end", "tool_execution_start", "tool_execution_end"])
    pi.on(type, (event: any) => send({ ...event, type, job: activeJob }));
  pi.on("agent_settled", () => { send({ type: "agent_settled", job: activeJob }); activeJob = undefined; });
  pi.on("session_shutdown", cleanup);
}
