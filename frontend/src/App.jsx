import { useState } from 'react';
import axios from 'axios';

// A component to display the results in a structured way
const ResultsDisplay = ({ data }) => {
  if (!data) return null;
  return (
    <div className="mt-6 p-4 bg-gray-50 rounded-lg border animate-fade-in">
      <h2 className="text-xl font-bold mb-4 text-gray-800">Analysis Results</h2>
      <div className="space-y-4">
        <div>
          <p className="font-semibold text-gray-700">Decision:
            <span className={`ml-2 px-3 py-1 text-sm font-bold rounded-full ${
              data.decision === 'approved' ? 'bg-green-100 text-green-800' :
              data.decision === 'rejected' ? 'bg-red-100 text-red-800' :
              'bg-yellow-100 text-yellow-800'
            }`}>
              {data.decision?.replace('_', ' ').toUpperCase()}
            </span>
          </p>
        </div>
        <div>
          <p className="font-semibold text-gray-700">Amount:</p>
          <p className="text-gray-900">{data.amount}</p>
        </div>
        <div>
          <p className="font-semibold text-gray-700">Justification:</p>
          <p className="text-gray-900">{data.justification}</p>
        </div>
        {data.matched_clauses && data.matched_clauses.length > 0 && (
          <div>
            <h3 className="font-semibold text-gray-700">Matched Clauses:</h3>
            <ul className="list-disc list-inside mt-2 space-y-3">
              {data.matched_clauses.map((clause, index) => (
                <li key={index} className="p-3 bg-gray-100 rounded-md border border-gray-200">
                  <p className="font-bold text-gray-800">{clause.clause_id || 'N/A'}</p>
                  <p className="text-sm text-gray-600">{clause.text}</p>
                  <p className="text-xs text-gray-400 mt-1">{clause.document}</p>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
};


function App() {
  const [file, setFile] = useState(null);
  const [query, setQuery] = useState("Is a 46M eligible for knee surgery in Pune if policy is 3 months old?");
  const [response, setResponse] = useState(null);
  const [status, setStatus] = useState('Ready. Please upload a document.');
  const [error, setError] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [isReadyForQuery, setIsReadyForQuery] = useState(false);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
        setFile(selectedFile);
        setStatus(`Ready to process '${selectedFile.name}'.`);
        setError('');
        setResponse(null);
        setIsReadyForQuery(false);
    }
  };

  const handleProcessDocument = async () => {
    if (!file) {
      setError('Please select a file first.');
      return;
    }

    const formData = new FormData();
    formData.append('file', file);

    setIsProcessing(true);
    setError('');
    setResponse(null);
    setIsReadyForQuery(false);


    try {
      setStatus('1/2: Uploading and processing document...');
      const uploadRes = await axios.post('/api/upload/', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setStatus('2/2: Embedding document... This may take a moment.');
      await axios.post('/api/embed/');
      setStatus(`Document '${file.name}' is processed and ready for queries.`);
      setIsReadyForQuery(true);

    } catch (err) {
      const errorMessage = err.response?.data?.detail || err.message;
      setError(`An error occurred: ${errorMessage}`);
      setStatus('Processing failed.');
      setIsReadyForQuery(false);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleQuerySubmit = async () => {
    if (!query) {
      setError('Please enter a query.');
      return;
    }

    setIsProcessing(true);
    setError('');
    setResponse(null);
    setStatus('Analyzing your query...');

    try {
      const queryRes = await axios.post('/api/query/', { query });
      setResponse(queryRes.data);
      setStatus('Analysis complete.');

    } catch (err) {
      const errorMessage = err.response?.data?.detail || err.message;
      setError(`An error occurred: ${errorMessage}`);
      setStatus('Query failed.');
    } finally {
      setIsProcessing(false);
    }
  };


  return (
    <div className="min-h-screen bg-gray-100 text-gray-800 flex items-center justify-center p-4 font-sans">
      <div className="w-full max-w-3xl mx-auto bg-white shadow-2xl rounded-2xl p-8">
        <h1 className="text-4xl font-bold text-center mb-6 text-gray-900">Insurance Document AI Analyst</h1>

        <div className="mb-8 p-4 border rounded-lg bg-gray-50">
          <label className="block text-lg font-semibold mb-2 text-gray-700" htmlFor="file_upload">
            Step 1: Upload & Process Policy
          </label>
          <div className="flex items-center space-x-4">
            <input
              type="file"
              id="file_upload"
              onChange={handleFileChange}
              className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100 cursor-pointer"
              accept=".pdf"
            />
            <button
              onClick={handleProcessDocument}
              disabled={!file || isProcessing}
              className="px-5 py-2 bg-blue-600 text-white font-semibold rounded-lg shadow-md hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors duration-300"
            >
              {isProcessing && status.includes('Embedding') ? 'Processing...' : 'Process'}
            </button>
          </div>
        </div>

        <div className="mb-4 p-4 border rounded-lg bg-gray-50">
          <label className="block text-lg font-semibold mb-2 text-gray-700" htmlFor="query_input">
            Step 2: Ask a Question
          </label>
          <textarea
            id="query_input"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            rows="3"
            className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 transition-shadow"
            placeholder="e.g., Is maternity cover included for a 30-year-old?"
          />
          <button
            onClick={handleQuerySubmit}
            disabled={isProcessing || !isReadyForQuery}
            className="mt-3 w-full px-4 py-3 bg-green-600 text-white font-bold rounded-lg shadow-md hover:bg-green-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors duration-300"
          >
            {isProcessing && status.includes('Analyzing') ? 'Analyzing...' : 'Get Analysis'}
          </button>
        </div>

        <div className="h-12 text-center flex items-center justify-center">
          {status && !error && <p className="text-gray-600 animate-pulse">{status}</p>}
          {error && <p className="text-red-600 font-semibold">{error}</p>}
        </div>

        <ResultsDisplay data={response} />
      </div>
    </div>
  );
}

export default App;
