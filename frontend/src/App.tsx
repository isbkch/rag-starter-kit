import React, { useState, useEffect, useCallback } from "react";
import type { SearchResult, Citation } from "./api";
import { search as searchApi, uploadDocument } from "./api";

function Search() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [searchType, setSearchType] = useState<"hybrid" | "vector" | "keyword">("hybrid");
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [searchInfo, setSearchInfo] = useState<{ total: number; time: number } | null>(null);

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
    
    try {
      const res = await searchApi(queryToSearch, searchType);
      setResults(res.results);
      setSearchInfo({ total: res.total_results, time: res.search_time });
    } catch {
      setError("Search failed. Please try again.");
      setResults([]);
      setSearchInfo(null);
    } finally {
      setLoading(false);
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
      
      {/* Search Type Selector */}
      <div className="mb-4">
        <label className="block text-sm font-medium text-gray-700 mb-2">Search Type</label>
        <div className="flex gap-2">
          {["hybrid", "vector", "keyword"].map((type) => (
            <button
              key={type}
              onClick={() => setSearchType(type as any)}
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
                Searching...
              </div>
            ) : (
              "Search"
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

        {searchInfo && (
          <div className="text-sm text-gray-600 mb-4">
            Found {searchInfo.total} results in {searchInfo.time.toFixed(2)}s using {searchType} search
          </div>
        )}
        
        {results.length === 0 && !loading && !error && query.trim() && (
          <div className="text-center py-8">
            <div className="text-gray-500 text-lg mb-2">No results found</div>
            <div className="text-gray-400 text-sm">Try adjusting your search terms or search type</div>
          </div>
        )}

        {results.length === 0 && !loading && !error && !query.trim() && (
          <div className="text-center py-12">
            <div className="text-gray-500 text-lg mb-2">Ready to search</div>
            <div className="text-gray-400 text-sm">Enter your question above to get started</div>
          </div>
        )}
        
        {results.length > 0 && (
          <div className="space-y-6">
            {results.map((result, index) => (
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
      <h2 className="text-2xl font-bold mb-4">Upload</h2>
      <form onSubmit={handleUpload} className="flex flex-col gap-4">
        <input
          type="file"
          onChange={handleFileChange}
          className="file:mr-4 file:py-2 file:px-4 file:rounded file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
        />
        <button
          type="submit"
          className="bg-blue-600 text-white px-4 py-2 rounded disabled:opacity-50"
          disabled={!file || loading}
        >
          {loading ? "Uploading..." : "Upload"}
        </button>
      </form>
      {error && <p className="mt-4 text-red-600">{error}</p>}
      {status && <p className="mt-4 text-green-600">{status}</p>}
    </div>
  );
}

export default function App() {
  const [page, setPage] = useState<'search' | 'upload'>('search');

  return (
    <div className="min-h-screen bg-gray-50">
      <nav className="bg-white shadow mb-8">
        <div className="max-w-2xl mx-auto px-4 py-3 flex gap-4">
          <button
            className={`font-semibold ${page === 'search' ? 'text-blue-600' : 'text-gray-700'}`}
            onClick={() => setPage('search')}
          >
            Search
          </button>
          <button
            className={`font-semibold ${page === 'upload' ? 'text-blue-600' : 'text-gray-700'}`}
            onClick={() => setPage('upload')}
          >
            Upload
          </button>
        </div>
      </nav>
      {page === 'search' ? <Search /> : <Upload />}
    </div>
  );
}
