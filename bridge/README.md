# Kindi Bridge

Kindi Bridge is a terminal workspace for agents that run on Kindi. It keeps the conversations, the scheduled
instructions, the completion checks, the local knowledge folders, the live Kindi data and the data of the computer
in one place. Version 0.6.0, for Kindi 0.5.0 and later.

**License: AGPL-3.0; commercial licenses available.** Kindi Bridge is free software under the
[GNU Affero General Public License v3.0](LICENSE), the same license as Kindi. For use in a closed product or a hosted
service, a paid commercial license is available. See [COMMERCIAL-LICENSE.md](COMMERCIAL-LICENSE.md).

The interface has the Kindi look: a black background, lime accents, conversations in the Kindi theme, rounded
panels, resource bars, live graphs and sparklines. A menu bar is always visible:
**Agents · Schedule · Kindi · Files · System · View · Help**. To open a menu, click a title, or press **Alt-M** or
**F10**. The arrow keys move through a menu. Enter selects an item. Esc closes the menu.

## Install with Kindi

```sh
curl -fsSL https://amirhusain.com/kindi/install.sh | bash -s -- --with-bridge
```

In a terminal, the Kindi installer asks if you want Bridge. `--with-bridge` installs it without the question.
`--no-bridge` does not install it. Bridge runs on the host, not in a container:

- the Python environment is in `~/.kindi/bridge`;
- the command is `~/.local/bin/kindi-bridge`;
- the state is in `~/.local/state/kindi-bridge`.

Bridge has its own pi folder there, so your pi setup does not change. The installer connects the agents of Bridge to
the first Kindi server. `install.sh --uninstall` also removes Bridge. With `--purge`, it can also delete the state.

Then run `kindi-bridge`. From another computer, run `ssh -t HOST kindi-bridge`.

## Connect to Kindi

**Kindi > Servers & features** shows each Kindi server that Bridge finds. These are the server that the agents use
now, the servers that the Kindi installer started (`~/.kindi/install.json`), and the providers in the pi
configuration of Bridge. Each row shows the model, the context, the Kindi version, the output head, the drafter
and the server state (busy or not). Below the servers, the screen lists the Kindi features, with the live value of
the current server when `/health` gives one.

- To use a server for new agents, select it and press **u**, or use **Kindi > Use a server for agents**.
- To add a server on another computer, use **Kindi > Connect to an address**.
- **Kindi > Full report** shows all the data as text.

## Start an agent

1. Select **Agents > New agent**.
2. Type the **Name** and the **Mission**. The form already contains the work folder.
3. Select **Create & Start**. Bridge sends the mission at once and opens the conversation.

The **Agents** menu shows each agent and its status. Select a name to go to that agent. **Agents > All agents &
status** shows a table: the name, the status, the instructions in the queue and the mission. Enter opens the
selected agent. You do not edit configuration files.

Type more instructions in the conversation. To send an instruction to one agent, select **Agents > Assign mission /
send prompt**. When an agent is busy, it gets the instructions in its queue at the end of its current turn. The
header shows the name of the selected agent, its status and the instructions that wait.

The conversation is the main pane. Next to it are a command pane and live side panels (the data of the computer and
of Kindi). In a form, Tab or Enter goes to the next field. To send the form, select its action button or press
Ctrl-S. Ctrl-E opens a long mission in the full text editor.

**View > Detach** closes the screen, and the agents continue to run. Each agent keeps the same session ID, so it
can continue its saved conversation when you start Bridge again.

## Screens and controls

| Menu | Items | Actions |
|---|---|---|
| Agents | Open conversation, New agent, Send an instruction, All agents, then each agent | Select an agent to open it. In All agents: Enter opens, `n` makes an agent, `m` sends a mission, `i` shows the details. |
| Schedule | Schedule a prompt, Check until done, Scheduled prompts, Activity log | Look at, delete (`d`; `u` undoes it) or pause (space) the instructions in the queue. |
| Kindi | Servers & features, Use a server for agents, Connect to an address, Full report | Enter shows the `/health` of a server; `u` uses it for new agents; `c` connects to an address; `i` shows the full report. |
| Files | Project files, Add a knowledge folder, Knowledge & search | Look at, preview and edit files (Ctrl-S saves, Esc cancels), see the Git changes; index, attach and search reference folders. |
| System | Performance, Processes, Listening ports, Cron & timers | Look at the resources and the ports, stop the process that owns a port, look at cron jobs and timers. |
| View | Command pane, Zoom / restore, Refit panels, Detach | Controls for the layout. |
| Help | Getting started, Keys and mouse, About Bridge | The first steps, all the keys, the version and the state folder. |

All menus, tables, forms and editors have the same colors, frames, spaces and visible commands.

- In a table, the selected row is lime. The arrow keys, Page Up, Page Down, Home and End move the selection. You can
  also select and scroll with the mouse. `/` filters the rows.
- **Esc** or **q** goes from a table back to the workspace.
- When the terminal has less than 100 columns or 28 rows, the content pane fills the screen. When the terminal is
  larger again, the side panels come back.
- The other panes, the conversations and the text that you typed do not change when you go to another view.

Over SSH, the menus that you can click and Alt-M prevent problems with the function keys. F10 also opens the menu.
Alt-A, Alt-S, Alt-K, Alt-F, Alt-Y, Alt-V and Alt-H open each menu. Alt-W goes back to the conversation. F1 to F9
(or Ctrl-b and then 1 to 9) open the screens: conversation, agents, scheduled prompts, Kindi, files, knowledge,
performance, processes and ports. Ctrl-b 0 goes back to the conversation.

The Processes and Ports screens read their data in background threads. The selection stays on the same process or
socket when the data refreshes. Before Bridge stops a process:

- it compares the start time of the process, so it never stops a new process that has the same PID;
- it does not stop its own control or session process;
- it shows the owner and asks for `y` before it sends SIGTERM or SIGKILL.

To close a port, Bridge stops the process that owns it. This can also close the other sockets of that process.
Bridge cannot close only one socket. A supervisor can start a stopped program again. When the owner is not known or
not accessible, Bridge shows it but does not offer to stop it.

## Install by hand

You need Linux, Python 3.10 or later, tmux 3.2 or later, and the pi coding agent 1.3.0 (Node.js 22.19 or later):

```sh
python3 -m venv .venv
.venv/bin/pip install -e '.[knowledge]'
.venv/bin/kindi-bridge
```

The base install (`pip install -e .`) gives agents, schedules, system controls and keyword search in plain text. The
`knowledge` extra adds text from PDF and Office files, and embeddings on the CPU. The first semantic index downloads
the small BGE embedding model. The FastEmbed [usage guide](https://qdrant.github.io/fastembed/) describes its
interface. After the download, all indexes and searches are local. Bridge does not send document text to an
embedding API.

Bridge finds pi on the PATH or in `~/.nvm/versions/node/*/bin/pi`. The native workbench loads a pi extension that
comes with Bridge, and a Python sidecar. A saved prompt goes into pi through its message API only when pi is idle,
has no work in its queue and its editor is empty. Bridge does not type keys into pi. Native pi keeps its full input,
its tool display and its extension dialogs. Agents that you made before the workbench existed can continue to use the
headless JSONL RPC worker. `kindi-bridge ui` is the separate table interface for diagnostics.

## Scheduled prompts

```sh
kindi-bridge agent add builder --dir ~/my-project --start
kindi-bridge jobs add 'Review the benchmark results' --at '16:30'
kindi-bridge jobs add 'Try the alternative approach' --agent builder --at 'in 20m'
kindi-bridge jobs add 'Report progress' --every 30m --limit 4
```

Without `--agent`, the prompt goes to the default agent. The time can be `now`, `in 20m`, `HH:MM` (the next time in
the time zone of the host), or an ISO time stamp with a UTC offset. The UI shows the time that Bridge calculated.
Bridge keeps the prompt text as you typed it. It adds retrieved context and check data as separate blocks.

At the scheduled time, the prompt is ready to send. Each worker sends one turn at a time. It records three states:
sending, accepted and completed. When an agent is not available, its late prompts wait. A repeated job sends one
prompt for the intervals that it missed, not one prompt for each interval. After a restart, a worker marks a delivery
that it cannot confirm as **uncertain**. You must look at it before you send it again:

```sh
kindi-bridge jobs history
kindi-bridge jobs retry JOB_ID
```

In the UI, `d` deletes future prompts and prompts that wait. You can undo this for ten seconds. Bridge cannot take
back a prompt that it started to send. A repeated job is one row, so a delete stops all its future prompts. Pause
also works only before the delivery.

## Completion checks

```sh
kindi-bridge jobs check --agent builder \
  --task 'Repair the importer: acceptance is the CSV regression suite passing' \
  --at 'in 20m' --every 30m --limit 6 \
  --deadline '2026-10-27T18:00:00-05:00'
```

Bridge sends: "Was the previous job successful? If not, plan an alternate strategy and resolve it." Each check
contains the task, the result of the last check and the format for the completion record. The checks continue when
you are away. They stop when:

- the model confirms that the task is complete;
- the limit or the deadline is reached;
- a delivery or provider error occurs.

Each rule has only one open check at a time. The limit counts the accepted checks. When the confirmation is missing
or not correct, Bridge does not count it as a success. The model is responsible for its evidence. Bridge does not run
its own verification commands yet.

Give `--task` the exact objective and the acceptance condition. Bridge does not find the task from other tools, and
it does not collect the files of an earlier task. It gives each check the result of the last check of the same rule.

## Knowledge folders

```sh
kindi-bridge knowledge add ~/reference-material --agent builder
kindi-bridge knowledge list
kindi-bridge knowledge search 'how do we restore a failed database?'
```

The background service finds and indexes all files in the folder and its subfolders. Then it looks for changes
every five minutes. `knowledge index COLLECTION_ID` indexes the folder again at once. `--keyword-only` adds a folder
without a model download.

File types:

- UTF-8 text, source code, Markdown, logs and CSV;
- the text of PDF files, with the page numbers;
- DOCX paragraphs and tables, PPTX text, tables and notes, XLSX cells and formulas.

Bridge keeps workbook formulas as text and never runs them. It marks PDF pages that have no text, for OCR at a
later time. The collection details show the files that Bridge cannot read: binary Office files, encrypted files,
other encodings and read errors. Bridge does not index files of more than 20 MiB, and it shows them. It does not
index hidden folders and files, symbolic links, credential files and the usual dependency and build folders.

Bridge mixes the SQLite FTS5 keyword rank and the rank of local BGE-small embeddings (reciprocal rank fusion). Each
passage keeps its place in the source file. An agent searches only the collection that you attached to it. Bridge
adds the results to the next prompt of the agent, as a reference block of 1,500 UTF-8 bytes or less. Bridge does
not count model tokens, and it does not change the earlier part of the pi conversation. Each index updates one
document at a time, in a transaction. When a file is deleted or cannot be read, its passages are removed from the
results. When the embeddings fail, the keyword search continues, and the collection shows a warning.

In this version, the vector search reads all embeddings of a collection in memory. It is for small local
collections, not for millions of passages. The paths are on the host that runs Bridge. Manual prompts and
scheduled prompts both get results from the collection of their agent. The quick editor in Files edits UTF-8 text
in the Kindi style. To use an external editor, set `$EDITOR`. When you save, Bridge compares the file with the
version that you opened. If the agent changed the file, Bridge keeps your edits in a conflict file.

## State and development

The state is in `~/.local/state/kindi-bridge`, out of source control. To use another folder, set
`BRIDGE_STATE=/path` or give `--state /path`. `BRIDGE_TMUX_SOCKET` gives a separate tmux server, for example for
tests. The UI starts one background service, with a kernel lock. Each agent worker also has its own lock, so only
one copy runs. SQLite uses WAL, and the job states stay on the disk.

[contrib/kindi-bridge.service](contrib/kindi-bridge.service) is an example systemd user unit for a service that runs
between logins. Bridge does not turn it on. After a reboot, Bridge does not start the agents again by itself. Start
the saved agents from the Agents menu. The service does not stop other agents and does not take their terminals.

```sh
python -m unittest discover -s tests -v
```

The tests include:

- the target of each prompt, exact prompts, and a delete at the same time as a delivery;
- completion evidence and limited retries;
- documents that change or are deleted, and read errors;
- real ports, protection against a reused PID, and the stop of child processes that Bridge owns;
- a real RPC subprocess, screen changes and terminal sizes in a PTY;
- Kindi server discovery and checks against a local HTTP server, and the pi provider that Bridge writes.

The tests never stop processes that are not theirs.

To test the native pi integration against a local HTTP server (no live model calls), run
`BRIDGE_TEST_PI=/path/to/pi python -m unittest discover -s tests -v`. This also tests the protection of the editor
text, the scheduled turns, the side panels across pages, and the narrow and wide layouts.

Limits of this version:

- The event feed is a list, not a dependency graph.
- The resource history is short and is not saved.
- Schedules and checks are exact; Bridge does not change their times by itself.
- Not done yet: priority and admission control for servers, control of agents from other tools, automatic
  verification commands, OCR and symbol navigation in the source index.

To move an old RPC agent to the native workbench, Bridge waits until the agent is idle. The earlier RPC conversation
files stay in their folder. Bridge does not merge them into the new native session.
