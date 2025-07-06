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
    stream: false
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

export interface StreamingSearchCallbacks {
  onMetadata?: (metadata: any) => void;
  onResult?: (result: SearchResult, index: number) => void;
  onSummary?: (summary: any) => void;
  onError?: (error: string) => void;
  onComplete?: () => void;
}

export async function searchStreaming(
  query: string,
  callbacks: StreamingSearchCallbacks,
  searchType: "hybrid" | "vector" | "keyword" = "hybrid",
  limit: number = MAX_SEARCH_RESULTS
): Promise<void> {
  const url = `${API_URL}/search/`;
  const payload = {
    query,
    search_type: searchType,
    limit,
    min_score: MIN_SEARCH_SCORE,
    stream: true
  };
  
  log('Starting streaming search with payload:', payload);
  
  try {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    
    if (!res.ok) {
      const error = await res.text();
      logError('Streaming search failed:', error);
      throw new Error(`Streaming search failed: ${res.status} ${res.statusText}`);
    }
    
    if (!res.body) {
      throw new Error('Response body is null');
    }
    
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    
    try {
      while (true) {
        const { done, value } = await reader.read();
        
        if (done) {
          break;
        }
        
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        
        // Process all complete lines
        for (let i = 0; i < lines.length - 1; i++) {
          const line = lines[i].trim();
          if (line) {
            try {
              const data = JSON.parse(line);
              
              switch (data.type) {
                case 'metadata':
                  log('Received metadata:', data);
                  callbacks.onMetadata?.(data);
                  break;
                case 'result':
                  log('Received result:', data.index, data.data);
                  callbacks.onResult?.(data.data, data.index);
                  break;
                case 'summary':
                  log('Received summary:', data);
                  callbacks.onSummary?.(data);
                  break;
                case 'error':
                  logError('Received error:', data.error);
                  callbacks.onError?.(data.error);
                  break;
                default:
                  log('Unknown message type:', data.type);
              }
            } catch (parseError) {
              logError('Error parsing streaming response:', parseError, 'Line:', line);
            }
          }
        }
        
        // Keep the last incomplete line in the buffer
        buffer = lines[lines.length - 1];
      }
      
      callbacks.onComplete?.();
      log('Streaming search completed');
      
    } finally {
      reader.releaseLock();
    }
    
  } catch (error) {
    logError('Streaming search error:', error);
    callbacks.onError?.(error instanceof Error ? error.message : String(error));
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
  const url = `${API_URL}/documents/upload`;
  const formData = new FormData();
  formData.append("file", file);
  
  log('Uploading document:', file.name, file.size);
  
  try {
    const res = await fetch(url, {
      method: "POST",
      body: formData,
    });
    
    if (!res.ok) {
      const error = await res.text();
      logError('Upload failed:', error);
      throw new Error(`Upload failed: ${res.status} ${res.statusText}`);
    }
    
    const result = await res.json();
    log('Upload successful:', result);
    return result;
  } catch (error) {
    logError('Upload error:', error);
    throw error;
  }
}

export interface Document {
  id: string;
  title: string;
  filename: string;
  file_size: number;
  document_type: string;
  status: string;
  total_chunks: number;
  processed_chunks: number;
  created_at: string;
  updated_at: string;
  processed_at?: string;
  error_message?: string;
}

export interface DocumentListResponse {
  documents: Document[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
}

export async function listDocuments(
  page: number = 1,
  pageSize: number = 10,
  status?: string
): Promise<DocumentListResponse> {
  const offset = (page - 1) * pageSize;
  let url = `${API_URL}/documents/?limit=${pageSize}&offset=${offset}`;
  
  if (status) {
    url += `&status=${status}`;
  }
  
  log('Listing documents:', { page, pageSize, status });
  
  try {
    const res = await fetch(url);
    
    if (!res.ok) {
      const error = await res.text();
      logError('List documents failed:', error);
      throw new Error(`List documents failed: ${res.status} ${res.statusText}`);
    }
    
    const result = await res.json();
    log('List documents successful:', result);
    return result;
  } catch (error) {
    logError('List documents error:', error);
    throw error;
  }
}

export async function getDocument(documentId: string): Promise<Document> {
  const url = `${API_URL}/documents/${documentId}`;
  
  log('Getting document:', documentId);
  
  try {
    const res = await fetch(url);
    
    if (!res.ok) {
      const error = await res.text();
      logError('Get document failed:', error);
      throw new Error(`Get document failed: ${res.status} ${res.statusText}`);
    }
    
    const result = await res.json();
    log('Get document successful:', result);
    return result;
  } catch (error) {
    logError('Get document error:', error);
    throw error;
  }
}

export async function deleteDocument(documentId: string): Promise<void> {
  const url = `${API_URL}/documents/${documentId}`;
  
  log('Deleting document:', documentId);
  
  try {
    const res = await fetch(url, {
      method: "DELETE",
    });
    
    if (!res.ok) {
      const error = await res.text();
      logError('Delete document failed:', error);
      throw new Error(`Delete document failed: ${res.status} ${res.statusText}`);
    }
    
    log('Delete document successful');
  } catch (error) {
    logError('Delete document error:', error);
    throw error;
  }
}

export async function reindexDocument(documentId: string): Promise<void> {
  const url = `${API_URL}/documents/${documentId}/reindex`;
  
  log('Reindexing document:', documentId);
  
  try {
    const res = await fetch(url, {
      method: "POST",
    });
    
    if (!res.ok) {
      const error = await res.text();
      logError('Reindex document failed:', error);
      throw new Error(`Reindex document failed: ${res.status} ${res.statusText}`);
    }
    
    log('Reindex document successful');
  } catch (error) {
    logError('Reindex document error:', error);
    throw error;
  }
}
