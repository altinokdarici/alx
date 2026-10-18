import { spawn } from 'node:child_process';
import process  from 'node:process';
import { createInterface } from 'node:readline';

const readLineInterface = createInterface({
	input: process.stdin,
	output: process.stdout
});

function askQuestion(question: string) {
	return new Promise((resolve) => readLineInterface.question(question, resolve));
}

type Message = {
	role: 'user' | 'assistant',
	content: string
};

async function request(messages: Message[]){
	const ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages";
	console.log('Making request to the LLM provider', messages);
	const response = await fetch(ANTHROPIC_API_URL, {
		headers: {
			'Content-Type': 'application/json',
			'anthropic-version': '2023-06-01',
			'X-Api-Key': process.env.CLAUDE_API_KEY
		},
		method: 'POST',
		body: JSON.stringify({
			model: 'claude-opus-5-5',
		        stream: false,
			max_tokens: 1024,
			system: [
				{
					type: 'text',
					text: 'You are a coding agent, you use bash tool to write code, read files, write files, execute code'
				}
			],
			messages,
			tools: [{
				input_schema: {
					type: 'object',
					properties: {
						input: { type: 'string'}
					},
					required: ['input']
				},
				name: 'bash'
			}]
		})
	});
	console.log('LLM response status', response.status);
	const data = await response.json();
	console.log('LLM response', data);
	return data; 
}

type BashResult = { stdout: string[], stderr: string[], exitCode: number};

function executeBash(input: string) : BashResult {
	return new Promise<BashResult>((resolve) => {
		const result: BashResult = { stdout: [], stderr: [], exitCode: 0};
		const bash = spawn('bash');
		bash.stdout.on('data', (data) => {
			result.stdout.push(data.toString());
		});
	
		bash.stderr.on('data', (data) => {
			result.stderr.push(data);
		});

		bash.on('close', (code) => {
			result.exitCode = code;
			resolve(result);
		});

		bash.stdin.write(input);
		bash.stdin.end();
	});
}

type ModelOutput = {
	id: string,
	stop_reason: string,
	content: {
		type: string,
		id: string,
		name: string,
		input: unknown
	}[]
}


async function main(){
	const messages: Message[] = [];
	console.log('Ask user input');
	const userMessage = await askQuestion('Its your turn');	
	messages.push({
		role: 'user',
		content: userMessage
	});
	

	let result = (await request(messages)) as ModelOutput;

	while(result.stop_reason === 'tool_use') {
		messages.push({ role: 'assistant', content: result.content });
		
		if(result.stop_reason !== 'tool_use'){
			break;
		}		
		
		for(const content of result.content) {
			if(content.type === 'tool_use') {
				console.log('Making bash tool call with', content);
				const toolResult = await executeBash(content.input.input);
				messages.push({
					role: 'user',
					content: [{
						type: 'tool_result',
						tool_use_id: content.id,
						content: JSON.stringify(toolResult)
					}]
				});
			}
		}
		result = await request(messages);	
	}
}	

await main();
