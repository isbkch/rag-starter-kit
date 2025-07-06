// API service for search and document upload

export interface Citation {
  id: number;
  ref: string;
  snippet: string;
}

export interface SearchResult {
  id: string;
  content: string;
  score: number;
  metadata: Record<string, unknown>;
  source: string;
  title: string;
  context?: string;
  citations: Citation[];
}

export interface SearchResponse {
  results: SearchResult[];
  total_results: number;
  search_time: number;
  search_type: string;
  metadata?: Record<string, unknown>;
}

export async function search(
  query: string,
  searchType: "hybrid" | "vector" | "keyword" = "hybrid"
): Promise<SearchResponse> {
  const res = await fetch("/api/v1/search/", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query,
      search_type: searchType,
      limit: 10,
      min_score: 0.0,
    }),
  });
  if (!res.ok) throw new Error("Search failed");
  return res.json();
}

export interface UploadResponse {
  document_id: string;
  filename: string;
  file_size: number;
  chunks_created: number;
  processing_time: number;
  status: string;
  message: string;
}

export async function uploadDocument(file: File): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch("/api/v1/documents/upload", {
    method: "POST",
    body: formData,
  });
  if (!res.ok) throw new Error("Upload failed");
  return res.json();
}
