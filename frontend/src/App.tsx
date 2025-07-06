import React from "react";

function App() {
  return (
    <div className="min-h-screen bg-gray-50 text-gray-900">
      <header className="bg-white shadow sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
          <h1 className="text-2xl font-bold tracking-tight text-blue-700">Enterprise RAG Platform</h1>
          <nav className="space-x-4">
            <a href="#search" className="text-gray-700 hover:text-blue-600 font-medium">Search</a>
            <a href="#upload" className="text-gray-700 hover:text-blue-600 font-medium">Upload</a>
            <a href="/docs" className="text-gray-400 hover:text-blue-400 font-medium" target="_blank" rel="noopener noreferrer">API Docs</a>
          </nav>
        </div>
      </header>
      <main className="max-w-4xl mx-auto px-4 py-10">
        {/* TODO: Route to Search and Upload pages */}
        <section id="search" className="mb-12">
          <h2 className="text-xl font-semibold mb-4">Semantic & Hybrid Search</h2>
          <div className="bg-white rounded-lg shadow p-6 min-h-[200px] flex items-center justify-center text-gray-400">
            Search UI coming soon...
          </div>
        </section>
        <section id="upload">
          <h2 className="text-xl font-semibold mb-4">Upload Documents</h2>
          <div className="bg-white rounded-lg shadow p-6 min-h-[150px] flex items-center justify-center text-gray-400">
            Upload UI coming soon...
          </div>
        </section>
      </main>
      <footer className="text-center text-gray-400 py-6 border-t mt-10 text-sm">
        &copy; {new Date().getFullYear()} Enterprise RAG Platform. All rights reserved.
      </footer>
    </div>
  );
}

export default App;
