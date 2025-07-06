// API service for search and document upload
import { API_URL, MAX_SEARCH_RESULTS, MIN_SEARCH_SCORE, log, logError } from './config';

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
  searchType: "hybrid" | "vector" | "keyword" = "hybrid",
  limit: number = MAX_SEARCH_RESULTS
): Promise<SearchResponse> {
  const url = `${API_URL}/search/`;
  const payload = {
    query,
    search_type: searchType,
    limit,
    min_score: MIN_SEARCH_SCORE,
  };
  
  log('Searching with payload:', payload);
  
  try {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    
    if (!res.ok) {
      const error = await res.text();
      logError('Search failed:', error);
      throw new Error(`Search failed: ${res.status} ${res.statusText}`);
    }
    
    const result = await res.json();
    log('Search successful:', result);
    return result;
  } catch (error) {
    logError('Search error:', error);
    throw error;
  }
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
