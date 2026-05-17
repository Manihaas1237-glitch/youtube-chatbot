About:
building a RAG system that can chat at real time abut the youtube videos.
for any questions reagrding a long YT videos, insgtead of spending time going through the entire video,you can ask this bot regarding that video or summarise the entire vide in 5 points etc



we need YT trancript. this can be done via langchain's YT loader package or YT API. We will via the YT API approach.Thiswill be document. This will be chunked into different parts and create embedding for them and store in vector store.by this our ending phase will be done.

for retrivers, we will use basic similarlity semantic search in vector store and retuen relevant documents.

Once we have relevant docs and query, we will make the prompt (this is augmentation) and send this prompt to LLM and retuen the response. This is generation.

lets build it in chain way..
1)for first chain,for prompt, we need question(from user) and context(we get this by question and retriver)
2)then second chain, prompt->llm->parser
