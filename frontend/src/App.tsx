import React, { useState } from "react";

// Define types for search results and citations
interface Citation {
  id: number;
  ref: string;
  snippet: string;
}

interface SearchResult {
  id: number;
  text: string;
  citations: Citation[];
}

function Search() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<SearchResult[]>([]); // Use explicit type

  // Placeholder search handler
  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    // TODO: Integrate with API
    setResults([
      {
        id: 1,
        text: "This is a sample result with a [1] citation.",
        citations: [
          { id: 1, ref: "Document 1", snippet: "Sample cited text." }
        ]
      }
    ]);
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
        <button className="bg-blue-600 text-white px-4 py-2 rounded" type="submit">
          Search
        </button>
      </form>
      <div>
        {results.length === 0 ? (
          <p className="text-gray-500">No results yet.</p>
        ) : (
          <ul className="space-y-4">
            {results.map(result => (
              <li key={result.id} className="border rounded p-4 bg-white shadow">
                {/* Simple citation highlighting: replace [n] with superscript */}
                <span dangerouslySetInnerHTML={{
                  __html: result.text.replace(/\[(\d+)\]/g, '<sup class="text-blue-600 cursor-pointer">[$1]</sup>')
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
  return (
    <div className="max-w-2xl mx-auto p-4">
      <h2 className="text-2xl font-bold mb-4">Upload</h2>
      {/* TODO: Implement upload UI */}
      <p className="text-gray-500">Upload functionality coming soon.</p>
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
