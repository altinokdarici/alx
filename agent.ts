import { spawn } from 'node:child_process';
import process from 'node:process';
import { createInterface } from 'node:readline';

const readLineInterface = createInterface({
	input: process.stdin,
	output: process.stdout
});

let inputClosed = false;
readLineInterface.on('close', () => { inputClosed = true; });

// Resolves to 'exit' on EOF (Ctrl+D / end of piped input) so the process can exit cleanly.
function askQuestion(question: string): Promise<string> {
	if (inputClosed) return Promise.resolve('exit');
	return new Promise((resolve) => {
		const onClose = () => resolve('exit');
		readLineInterface.once('close', onClose);
		readLineInterface.question(question, (answer) => {
			readLineInterface.off('close', onClose);
			resolve(answer);
		});
	});
}

type TextBlock = { type: 'text', text: string };
type ToolUseBlock = { type: 'tool_use', id: string, name: string, input: { input?: string } };
type ToolResultBlock = { type: 'tool_result', tool_use_id: string, content: string, is_error?: boolean };
type ContentBlock = TextBlock | ToolUseBlock | ToolResultBlock;

type Message = {
	role: 'user' | 'assistant',
	content: string | ContentBlock[]
};

type ModelOutput = {
	id: string,
	stop_reason: string,
	content: ContentBlock[]
};

async function request(messages: Message[]): Promise<ModelOutput> {
	const ANTHROPIC_API_URL = 'https://api.anthropic.com/v1/messages';
	const apiKey = process.env.CLAUDE_API_KEY;
	if (!apiKey) {
		throw new Error('CLAUDE_API_KEY environment variable is not set');
	}
	console.log('Making request to the LLM provider');
	const response = await fetch(ANTHROPIC_API_URL, {
		headers: {
			'Content-Type': 'application/json',
			'anthropic-version': '2023-06-01',
			'X-Api-Key': apiKey
		},
		method: 'POST',
		body: JSON.stringify({
			model: 'claude-opus-5-5',
			stream: false,
			max_tokens: 10240,
			system: [
				{
					type: 'text',
					text: 'You are a coding agent, you use bash tool to write code, read files, write files, execute code'
				}
			],
			messages,
			tools: [{
				name: 'bash',
				description: 'Run a bash script. The `input` string is piped to bash via stdin. Returns stdout, stderr and exit code.',
				input_schema: {
					type: 'object',
					properties: {
						input: { type: 'string', description: 'The bash script to execute' }
					},
					required: ['input']
				}
			}]
		})
	});
	const data = await response.json();
	if (!response.ok) {
		throw new Error(`LLM request failed (${response.status}): ${JSON.stringify(data)}`);
	}
	return data as ModelOutput;
}

type BashResult = { stdout: string, stderr: string, exitCode: number | null };

function executeBash(input: string): Promise<BashResult> {
	return new Promise<BashResult>((resolve) => {
		const stdout: string[] = [];
		const stderr: string[] = [];
		const bash = spawn('bash');
		bash.stdout.on('data', (data) => stdout.push(data.toString()));
		bash.stderr.on('data', (data) => stderr.push(data.toString()));
		bash.on('error', (err) => {
			resolve({ stdout: stdout.join(''), stderr: stderr.join('') + String(err), exitCode: -1 });
		});
		bash.on('close', (code) => {
			resolve({ stdout: stdout.join(''), stderr: stderr.join(''), exitCode: code });
		});
		bash.stdin.write(input);
		bash.stdin.end();
	});
}

function printText(output: ModelOutput) {
	for (const block of output.content) {
		if (block.type === 'text') {
			console.log(`\nAssistant: ${block.text}\n`);
		}
	}
}

// Process exit codes. `end_turn` is the only "successful" way for a model turn to finish.
const ExitCode = {
	end_turn: 0,
	max_tokens: 2,
	stop_sequence: 3,
	refusal: 4,
	pause_turn: 5,
	unknown: 1
} as const;

function exitCodeFor(stopReason: string): number {
	return ExitCode[stopReason as keyof typeof ExitCode] ?? ExitCode.unknown;
}

async function main() {
	const messages: Message[] = [];

	while (true) {
		const userMessage = (await askQuestion("It's your turn (type 'exit' to quit): ")).trim();
		if (userMessage === 'exit' || userMessage === 'quit') break;
		if (!userMessage) continue;

		messages.push({ role: 'user', content: userMessage });

		let result = await request(messages);
		messages.push({ role: 'assistant', content: result.content });
		printText(result);

		while (result.stop_reason === 'tool_use') {
			// All tool results must go in a single user message following the assistant turn.
			const toolResults: ToolResultBlock[] = [];
			for (const block of result.content) {
				if (block.type !== 'tool_use') continue;
				if (block.name !== 'bash' || typeof block.input?.input !== 'string') {
					toolResults.push({
						type: 'tool_result',
						tool_use_id: block.id,
						content: `Invalid tool call: ${block.name}`,
						is_error: true
					});
					continue;
				}
				console.log('$', block.input.input);
				const toolResult = await executeBash(block.input.input);
				console.log(toolResult);
				toolResults.push({
					type: 'tool_result',
					tool_use_id: block.id,
					content: JSON.stringify(toolResult)
				});
			}
			messages.push({ role: 'user', content: toolResults });

			result = await request(messages);
			messages.push({ role: 'assistant', content: result.content });
			printText(result);
		}

		if (result.stop_reason !== 'end_turn') {
			const code = exitCodeFor(result.stop_reason);
			console.error(`Unexpected stop_reason "${result.stop_reason}", exiting with code ${code}`);
			readLineInterface.close();
			process.exit(code);
		}
	}

	readLineInterface.close();
	process.exit(ExitCode.end_turn);
}

await main();
