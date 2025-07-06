import React, { useState, useEffect, useCallback } from "react";
import type { SearchResult, Citation, Document } from "./api";
import { search as searchApi, uploadDocument, searchStreaming, listDocuments, deleteDocument, reindexDocument } from "./api";

function Search() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [searchType, setSearchType] = useState<"hybrid" | "vector" | "keyword">("hybrid");
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [searchInfo, setSearchInfo] = useState<{ total: number; time: number } | null>(null);
  const [useStreaming, setUseStreaming] = useState(false);
  const [streamingStatus, setStreamingStatus] = useState<string>("");
  const [streamingResults, setStreamingResults] = useState<SearchResult[]>([]);

  // Debounced search for suggestions
  const debouncedGetSuggestions = useCallback(
    (searchQuery: string) => {
      const timeoutId = setTimeout(async () => {
        if (searchQuery.length > 2) {
          try {
            const response = await fetch(`/api/v1/search/suggestions?q=${encodeURIComponent(searchQuery)}&limit=5`);
            if (response.ok) {
              const data = await response.json();
              setSuggestions(data.suggestions || []);
              setShowSuggestions(true);
            }
          } catch (error) {
            console.error("Failed to get suggestions:", error);
          }
        } else {
          setSuggestions([]);
          setShowSuggestions(false);
        }
      }, 300);

      return () => clearTimeout(timeoutId);
    },
    []
  );

  // Effect for getting search suggestions
  useEffect(() => {
    const cleanup = debouncedGetSuggestions(query);
    return cleanup;
  }, [query, debouncedGetSuggestions]);

  const handleSearch = async (searchQuery?: string) => {
    const queryToSearch = searchQuery || query;
    if (!queryToSearch.trim()) return;

    setLoading(true);
    setError(null);
    setShowSuggestions(false);
    setStreamingStatus("");
    
    if (useStreaming) {
      setStreamingResults([]);
      setResults([]);
      setSearchInfo(null);
      
      try {
        await searchStreaming(queryToSearch, {
          onMetadata: (metadata) => {
            setStreamingStatus(`Starting ${metadata.search_type} search for "${metadata.query}"`);
          },
          onResult: (result, index) => {
            setStreamingStatus(`Received result ${index + 1}`);
            setStreamingResults(prev => [...prev, result]);
          },
          onSummary: (summary) => {
            setSearchInfo({ total: summary.total_results, time: summary.search_time });
            setStreamingStatus("Search completed");
          },
          onError: (error) => {
            setError(`Streaming search error: ${error}`);
          },
          onComplete: () => {
            setLoading(false);
            setStreamingStatus("");
          }
        }, searchType);
      } catch (err) {
        setError("Streaming search failed. Please try again.");
        setLoading(false);
        setStreamingStatus("");
      }
    } else {
      try {
        const res = await searchApi(queryToSearch, searchType);
        setResults(res.results);
        setStreamingResults([]);
        setSearchInfo({ total: res.total_results, time: res.search_time });
      } catch {
        setError("Search failed. Please try again.");
        setResults([]);
        setStreamingResults([]);
        setSearchInfo(null);
      } finally {
        setLoading(false);
      }
    }
  };

  const handleFormSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    handleSearch();
  };

  const handleSuggestionClick = (suggestion: string) => {
    setQuery(suggestion);
    setShowSuggestions(false);
    handleSearch(suggestion);
  };

  const highlightCitations = (content: string) => {
    return content.replace(
      /\[(\d+)\]/g, 
      '<sup class="text-blue-600 cursor-pointer font-semibold hover:text-blue-800 transition-colors" title="Click to view citation">[$1]</sup>'
    );
  };

  return (
    <div className="max-w-4xl mx-auto p-4">
      <h2 className="text-3xl font-bold mb-6 text-gray-800">RAG Search</h2>
      
      {/* Search Configuration */}
      <div className="mb-4 space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">Search Type</label>
          <div className="flex gap-2">
            {["hybrid", "vector", "keyword"].map((type) => (
              <button
                key={type}
                onClick={() => setSearchType(type as "hybrid" | "vector" | "keyword")}
                className={`px-3 py-1 rounded text-sm font-medium transition-colors ${
                  searchType === type
                    ? "bg-blue-600 text-white"
                    : "bg-gray-200 text-gray-700 hover:bg-gray-300"
                }`}
              >
                {type.charAt(0).toUpperCase() + type.slice(1)}
              </button>
            ))}
          </div>
        </div>
        
        <div className="flex items-center gap-2">
          <input
            id="streaming-toggle"
            type="checkbox"
            checked={useStreaming}
            onChange={(e) => setUseStreaming(e.target.checked)}
            className="w-4 h-4 text-blue-600 bg-gray-100 border-gray-300 rounded focus:ring-blue-500 focus:ring-2"
          />
          <label htmlFor="streaming-toggle" className="text-sm font-medium text-gray-700">
            Enable streaming search (results appear as they arrive)
          </label>
        </div>
      </div>

      {/* Search Form */}
      <div className="relative mb-6">
        <form onSubmit={handleFormSubmit} className="flex gap-2">
          <div className="flex-1 relative">
            <input
              className="w-full border border-gray-300 rounded-lg px-4 py-3 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              type="text"
              placeholder="Ask me anything about your documents..."
              value={query}
              onChange={e => setQuery(e.target.value)}
              onFocus={() => query.length > 2 && setShowSuggestions(true)}
              onBlur={() => setTimeout(() => setShowSuggestions(false), 200)}
            />
            
            {/* Search Suggestions */}
            {showSuggestions && suggestions.length > 0 && (
              <div className="absolute top-full left-0 right-0 bg-white border border-gray-300 rounded-lg shadow-lg z-10 mt-1">
                {suggestions.map((suggestion, index) => (
                  <button
                    key={index}
                    onClick={() => handleSuggestionClick(suggestion)}
                    className="w-full text-left px-4 py-2 hover:bg-gray-100 border-b border-gray-100 last:border-b-0 first:rounded-t-lg last:rounded-b-lg"
                  >
                    {suggestion}
                  </button>
                ))}
              </div>
            )}
          </div>
          
          <button 
            className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-lg font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed" 
            type="submit" 
            disabled={loading || !query.trim()}
          >
            {loading ? (
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                {useStreaming ? "Streaming..." : "Searching..."}
              </div>
            ) : (
              useStreaming ? "Stream Search" : "Search"
            )}
          </button>
        </form>
      </div>

      {/* Search Results */}
      <div>
        {error && (
          <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg mb-4">
            {error}
          </div>
        )}

        {streamingStatus && (
          <div className="bg-blue-50 border border-blue-200 text-blue-700 px-4 py-3 rounded-lg mb-4">
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 border-2 border-blue-600 border-t-transparent rounded-full animate-spin"></div>
              {streamingStatus}
            </div>
          </div>
        )}

        {searchInfo && (
          <div className="text-sm text-gray-600 mb-4">
            Found {searchInfo.total} results in {searchInfo.time.toFixed(2)}s using {searchType} search
            {useStreaming && " (streamed)"}
          </div>
        )}
        
        {(results.length === 0 && streamingResults.length === 0) && !loading && !error && query.trim() && (
          <div className="text-center py-8">
            <div className="text-gray-500 text-lg mb-2">No results found</div>
            <div className="text-gray-400 text-sm">Try adjusting your search terms or search type</div>
          </div>
        )}

        {(results.length === 0 && streamingResults.length === 0) && !loading && !error && !query.trim() && (
          <div className="text-center py-12">
            <div className="text-gray-500 text-lg mb-2">Ready to search</div>
            <div className="text-gray-400 text-sm">Enter your question above to get started</div>
          </div>
        )}
        
        {(results.length > 0 || streamingResults.length > 0) && (
          <div className="space-y-6">
            {(useStreaming ? streamingResults : results).map((result, index) => (
              <div key={result.id} className="bg-white border border-gray-200 rounded-lg p-6 shadow-sm hover:shadow-md transition-shadow">
                {/* Result Header */}
                <div className="flex items-start justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <span className="bg-blue-100 text-blue-800 text-xs font-medium px-2 py-1 rounded">
                      #{index + 1}
                    </span>
                    {result.title && (
                      <h3 className="font-semibold text-gray-800">{result.title}</h3>
                    )}
                  </div>
                  <div className="text-sm text-gray-500">
                    Score: {(result.score * 100).toFixed(1)}%
                  </div>
                </div>

                {/* Content with Citation Highlighting */}
                <div 
                  className="text-gray-700 leading-relaxed mb-4"
                  dangerouslySetInnerHTML={{
                    __html: highlightCitations(result.content)
                  }} 
                />

                {/* Citations */}
                {result.citations && result.citations.length > 0 && (
                  <div className="border-t border-gray-100 pt-4">
                    <h4 className="text-sm font-medium text-gray-600 mb-2">Sources:</h4>
                    <div className="space-y-2">
                      {result.citations.map((citation: Citation) => (
                        <div key={citation.id} className="text-sm">
                          <span className="font-semibold text-blue-600">[{citation.id}]</span>
                          <span className="text-gray-600 ml-2">{citation.ref}:</span>
                          <span className="text-gray-500 italic ml-2">"{citation.snippet}"</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Metadata */}
                {result.source && (
                  <div className="text-xs text-gray-400 mt-2">
                    Source: {result.source}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function Upload() {
  const [file, setFile] = useState<File | null>(null);
  const [status, setStatus] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setFile(e.target.files[0]);
      setStatus("");
      setError(null);
    }
  };

  const handleUpload = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) {
      setStatus("");
      setError("Please select a file to upload.");
      return;
    }
    setLoading(true);
    setError(null);
    setStatus("");
    try {
      const res = await uploadDocument(file);
      setStatus(res.message || `Successfully uploaded: ${file.name}`);
      setFile(null);
    } catch {
      setError("Upload failed. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-2xl mx-auto p-4">
      <h2 className="text-2xl font-bold mb-4">Upload Document</h2>
      <form onSubmit={handleUpload} className="flex flex-col gap-4">
        <div className="border-2 border-dashed border-gray-300 rounded-lg p-6">
          <input
            type="file"
            onChange={handleFileChange}
            accept=".pdf,.docx,.md,.txt"
            className="file:mr-4 file:py-2 file:px-4 file:rounded file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
          />
          <p className="text-sm text-gray-500 mt-2">
            Supported formats: PDF, DOCX, Markdown, Text (max 50MB)
          </p>
        </div>
        <button
          type="submit"
          className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded font-medium transition-colors disabled:opacity-50"
          disabled={!file || loading}
        >
          {loading ? "Uploading..." : "Upload Document"}
        </button>
      </form>
      {error && (
        <div className="mt-4 p-3 bg-red-50 border border-red-200 text-red-700 rounded">
          {error}
        </div>
      )}
      {status && (
        <div className="mt-4 p-3 bg-green-50 border border-green-200 text-green-700 rounded">
          {status}
        </div>
      )}
    </div>
  );
}

function Documents() {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(0);
  const [total, setTotal] = useState(0);
  const [statusFilter, setStatusFilter] = useState<string>("");
  const [actionLoading, setActionLoading] = useState<{ [key: string]: boolean }>({});

  const loadDocuments = useCallback(async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await listDocuments(page, 10, statusFilter || undefined);
      setDocuments(response.documents);
      setTotal(response.total);
      setTotalPages(response.total_pages);
    } catch (err) {
      setError("Failed to load documents. Please try again.");
    } finally {
      setLoading(false);
    }
  }, [page, statusFilter]);

  useEffect(() => {
    loadDocuments();
  }, [loadDocuments]);

  const handleDelete = async (documentId: string) => {
    if (!confirm("Are you sure you want to delete this document? This action cannot be undone.")) {
      return;
    }
    
    setActionLoading(prev => ({ ...prev, [documentId]: true }));
    
    try {
      await deleteDocument(documentId);
      await loadDocuments(); // Refresh the list
    } catch (err) {
      setError("Failed to delete document. Please try again.");
    } finally {
      setActionLoading(prev => ({ ...prev, [documentId]: false }));
    }
  };

  const handleReindex = async (documentId: string) => {
    setActionLoading(prev => ({ ...prev, [`reindex_${documentId}`]: true }));
    
    try {
      await reindexDocument(documentId);
      setError(null);
      // You might want to show a success message here
    } catch (err) {
      setError("Failed to reindex document. Please try again.");
    } finally {
      setActionLoading(prev => ({ ...prev, [`reindex_${documentId}`]: false }));
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'bg-green-100 text-green-800';
      case 'processing': return 'bg-blue-100 text-blue-800';
      case 'failed': return 'bg-red-100 text-red-800';
      case 'pending': return 'bg-yellow-100 text-yellow-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  return (
    <div className="max-w-6xl mx-auto p-4">
      <h2 className="text-3xl font-bold mb-6 text-gray-800">Document Management</h2>
      
      {/* Filters */}
      <div className="mb-6 flex gap-4 items-center">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">Filter by Status</label>
          <select
            value={statusFilter}
            onChange={(e) => {
              setStatusFilter(e.target.value);
              setPage(1);
            }}
            className="border border-gray-300 rounded-lg px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          >
            <option value="">All</option>
            <option value="completed">Completed</option>
            <option value="processing">Processing</option>
            <option value="failed">Failed</option>
            <option value="pending">Pending</option>
          </select>
        </div>
        
        <button
          onClick={loadDocuments}
          className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded font-medium transition-colors"
        >
          Refresh
        </button>
      </div>

      {error && (
        <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg mb-4">
          {error}
        </div>
      )}

      {/* Summary */}
      <div className="bg-white rounded-lg shadow p-4 mb-6">
        <div className="text-sm text-gray-600">
          Showing {documents.length} of {total} documents
          {statusFilter && ` (filtered by: ${statusFilter})`}
        </div>
      </div>

      {/* Documents List */}
      {loading ? (
        <div className="text-center py-8">
          <div className="inline-flex items-center gap-2">
            <div className="w-6 h-6 border-2 border-blue-600 border-t-transparent rounded-full animate-spin"></div>
            Loading documents...
          </div>
        </div>
      ) : documents.length === 0 ? (
        <div className="text-center py-12">
          <div className="text-gray-500 text-lg mb-2">No documents found</div>
          <div className="text-gray-400 text-sm">
            {statusFilter ? 'Try changing the status filter or upload some documents' : 'Upload some documents to get started'}
          </div>
        </div>
      ) : (
        <div className="space-y-4">
          {documents.map((doc) => (
            <div key={doc.id} className="bg-white border border-gray-200 rounded-lg p-6 shadow-sm">
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="flex items-center gap-3 mb-2">
                    <h3 className="font-semibold text-gray-800">{doc.title}</h3>
                    <span className={`px-2 py-1 rounded text-xs font-medium ${getStatusColor(doc.status)}`}>
                      {doc.status}
                    </span>
                  </div>
                  
                  <div className="grid grid-cols-2 gap-4 text-sm text-gray-600 mb-3">
                    <div><span className="font-medium">Filename:</span> {doc.filename}</div>
                    <div><span className="font-medium">Type:</span> {doc.document_type}</div>
                    <div><span className="font-medium">Size:</span> {formatFileSize(doc.file_size)}</div>
                    <div><span className="font-medium">Chunks:</span> {doc.processed_chunks}/{doc.total_chunks}</div>
                    <div><span className="font-medium">Created:</span> {new Date(doc.created_at).toLocaleDateString()}</div>
                    <div><span className="font-medium">Updated:</span> {new Date(doc.updated_at).toLocaleDateString()}</div>
                  </div>

                  {doc.error_message && (
                    <div className="text-sm text-red-600 bg-red-50 p-2 rounded mb-3">
                      <span className="font-medium">Error:</span> {doc.error_message}
                    </div>
                  )}
                </div>
                
                <div className="flex gap-2 ml-4">
                  <button
                    onClick={() => handleReindex(doc.id)}
                    disabled={actionLoading[`reindex_${doc.id}`] || doc.status === 'processing'}
                    className="bg-blue-600 hover:bg-blue-700 text-white px-3 py-1 rounded text-sm font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {actionLoading[`reindex_${doc.id}`] ? 'Reindexing...' : 'Reindex'}
                  </button>
                  
                  <button
                    onClick={() => handleDelete(doc.id)}
                    disabled={actionLoading[doc.id]}
                    className="bg-red-600 hover:bg-red-700 text-white px-3 py-1 rounded text-sm font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {actionLoading[doc.id] ? 'Deleting...' : 'Delete'}
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex justify-center items-center gap-2 mt-6">
          <button
            onClick={() => setPage(p => Math.max(1, p - 1))}
            disabled={page === 1}
            className="px-3 py-2 border border-gray-300 rounded text-sm disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-50"
          >
            Previous
          </button>
          
          <span className="px-3 py-2 text-sm text-gray-600">
            Page {page} of {totalPages}
          </span>
          
          <button
            onClick={() => setPage(p => Math.min(totalPages, p + 1))}
            disabled={page === totalPages}
            className="px-3 py-2 border border-gray-300 rounded text-sm disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-50"
          >
            Next
          </button>
        </div>
      )}
    </div>
  );
}

export default function App() {
  const [page, setPage] = useState<'search' | 'upload' | 'documents'>('search');

  const renderPage = () => {
    switch (page) {
      case 'search':
        return <Search />;
      case 'upload':
        return <Upload />;
      case 'documents':
        return <Documents />;
      default:
        return <Search />;
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Navigation Header */}
      <nav className="bg-white shadow-md mb-8">
        <div className="max-w-7xl mx-auto px-4">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center gap-8">
              <div className="flex-shrink-0">
                <h1 className="text-xl font-bold text-gray-900">Enterprise RAG Platform</h1>
              </div>
              
              <div className="flex gap-1">
                {[
                  { id: 'search', label: 'Search', icon: '🔍' },
                  { id: 'upload', label: 'Upload', icon: '📁' },
                  { id: 'documents', label: 'Documents', icon: '📄' }
                ].map((tab) => (
                  <button
                    key={tab.id}
                    onClick={() => setPage(tab.id as 'search' | 'upload' | 'documents')}
                    className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                      page === tab.id
                        ? 'bg-blue-100 text-blue-700 border-b-2 border-blue-600'
                        : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
                    }`}
                  >
                    <span className="mr-2">{tab.icon}</span>
                    {tab.label}
                  </button>
                ))}
              </div>
            </div>
            
            <div className="text-sm text-gray-500">
              v0.1.0
            </div>
          </div>
        </div>
      </nav>
      
      {/* Page Content */}
      <main>
        {renderPage()}
      </main>
    </div>
  );
}
