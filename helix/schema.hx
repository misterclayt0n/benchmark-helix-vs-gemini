N::DocumentChunk {
  doc_id: String,
  chunk_id: String,
  text: String
}

V::DocumentChunkEmbedding {
  embedding: [F64]
}

E::Chunk_to_Embedding {
  From: DocumentChunk,
  To: DocumentChunkEmbedding
}
