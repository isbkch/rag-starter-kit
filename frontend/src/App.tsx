import React, { useState } from "react";
import type { SearchResult, Citation } from "./api";
import { search as searchApi, uploadDocument } from "./api";

function Search() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const res = await searchApi(query);
      setResults(res.results);
    } catch {
      setError("Search failed. Please try again.");
      setResults([]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-2xl mx-auto p-4">
      <h2 className="text-2xl font-bold mb-4">Search</h2>
      <form onSubmit={handleSearch} className="flex gap-2 mb-6">
        <input
          className="flex-1 border rounded px-3 py-2"
          type="text"
          placeholder="Enter your query..."
          value={query}
          onChange={e => setQuery(e.target.value)}
        />
        <button className="bg-blue-600 text-white px-4 py-2 rounded" type="submit" disabled={loading}>
          {loading ? "Searching..." : "Search"}
        </button>
      </form>
      <div>
        {error && <p className="text-red-600 mb-2">{error}</p>}
        {results.length === 0 && !loading && !error ? (
          <p className="text-gray-500">No results yet.</p>
        ) : (
          <ul className="space-y-4">
            {results.map(result => (
              <li key={result.id} className="border rounded p-4 bg-white shadow">
                {/* Simple citation highlighting: replace [n] with superscript */}
                <span dangerouslySetInnerHTML={{
                  __html: result.content.replace(/\[(\d+)\]/g, '<sup class="text-blue-600 cursor-pointer">[$1]</sup>')
                }} />
                <div className="mt-2 text-sm text-gray-600">
                  {result.citations.map((c: Citation) => (
                    <div key={c.id}>
                      <span className="font-semibold">[{c.id}]</span> {c.ref}: <span className="italic">{c.snippet}</span>
                    </div>
                  ))}
                </div>
              </li>
            ))}
          </ul>
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
