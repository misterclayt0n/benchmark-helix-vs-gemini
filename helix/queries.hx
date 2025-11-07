QUERY insert_document_chunks(doc_id: String, chunk_id: String, text: String, embedding: [F64]) =>
  chunk <- AddN<DocumentChunk>({
    doc_id: doc_id,
    chunk_id: chunk_id,
    text: text
  })
  vec <- AddV<DocumentChunkEmbedding>(embedding)
  edge <- AddE<Chunk_to_Embedding>::From(chunk)::To(vec)
  RETURN chunk

QUERY search_chunks(embedding: [F64], top_k: I64) =>
  vecs <- SearchV<DocumentChunkEmbedding>(embedding, top_k)
  chunks <- vecs::In<Chunk_to_Embedding>
  RETURN chunks::{doc_id, chunk_id, text}

