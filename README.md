A three-fold architecture to assume diverse roles for LLAMA-3.2B in order to solve the language barrier problem,
generate answers in local language (Urdu), and only cater Pakistan's Tax Ordinance. Now we achieved this first by 
creating a RAG pipeline, where we used mini-LLM to generate embeddings for our documents, we had two docs, 1 referred 
to 800 pages of laws governing taxation, other referred to different combination of designations requiring different slabs of tax computation. 
Now we stored chunks + embeddings in a VectorDB and upon query of user, extracted the relevant context through similarity search, after that we used LLAMA-3.2-11B
to match the context with the query and create keyword specific tags, that would help streamline the prompt further ahead on basis of user's query and knowledge base. 
Then we sent those tags + query + context to LLAMA again to generate the relative pointers to each tag in a detailed simplified manner according to the context. 
Then we sent this generated text to LLAMA acting as a language converter, to prompt for Urdu Answers such that it can be understood by the majority of native population.
