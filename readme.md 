# KPI FICE RAG CHAT-BOT

#### To start the bot:
1. Unzip ***fiot_files.zip*** file in **data** dir, so faculty documents will be placed under ***data/fiot_files*** dir, you can also add another files with faculty info in *txt, pdf, docx, markdown & html* formats;
2. Create new telegram bot via **@BotFather** and update `TELEGRAM_TOKEN` value in `.env` file with a new bot's token;
3. Download [Ollama](https://ollama.com/download) to run free LLMs locally;
4. Run in terminal: `ollama run llama3.1:8b` - download & start llama3.1 model with 8 billion params locally, which requires ***8GM of RAM***. You can also choose other model from [Ollama](https://ollama.com/search) site, then replace `LLM_MODEL` name inside `.env` file;
5. Run `pip install -r requirements.txt` inside working directory to download all dependencies;
7. Run `python telegram_bot.py` in terminal.

####
Now you can go to your *telegram bot* and ask it questions about ***FICE :)***


###
#### Troubleshooting:
- if you don't have Nvidia GPU remove line `device = ComponentDevice.from_str("cuda")` & `device` variable usage from ***qa_pipeline.py*** and ***data_pipeline.py*** files)

###
#### Improve Bot's Answers:
##### To improve bot's answers you should use "stronger" LLM. To run this bot with OpenAI (ChatGPT) models:
1. Generate new OpenAI api key [here](https://platform.openai.com/api-keys) (you should also have credits on the account);
2. Update `OPENAI_API_KEY` in `.env` file with your OpenAI api key;
3. Go to `qa_pipeline.py` file and uncomment `qa_pipeline.add_component("llm", OpenAIChatGenerator())` line, then comment `qa_pipeline.add_component("llm", OllamaChatGenerator(...))` lines;
4. Re-start the bot: `python telegram_bot.py`.

###
#### Check *qa_pipeline.png and data_pipeline.png*: pipelines visualization.
