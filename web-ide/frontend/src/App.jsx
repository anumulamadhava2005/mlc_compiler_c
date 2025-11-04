import { useState, useEffect } from 'react'
import Editor from '@monaco-editor/react'
import { PlayCircle, Download, FileCode, Terminal, Loader2, CheckCircle, XCircle, FileText, Folder, FlaskConical, Upload, Send } from 'lucide-react'
import axios from 'axios'

const DEFAULT_CODE = `dataset "/home/madhava/datasets/flowers"

model ResNet50 {
    epochs = 10
    batch_size = 32
    learning_rate = 0.001
}`

function App() {
  const [code, setCode] = useState(DEFAULT_CODE)
  const [output, setOutput] = useState('')
  const [isCompiling, setIsCompiling] = useState(false)
  const [compilationStatus, setCompilationStatus] = useState(null)
  const [generatedFiles, setGeneratedFiles] = useState(null)
  const [generatedFileContent, setGeneratedFileContent] = useState(null)
  const [showPredictPanel, setShowPredictPanel] = useState(false)
  const [modelExists, setModelExists] = useState(false)
  const [predictionResult, setPredictionResult] = useState(null)
  const [isPredicting, setIsPredicting] = useState(false)
  const [predictionMode, setPredictionMode] = useState('csv') // 'csv' or 'manual'
  const [manualInput, setManualInput] = useState('')

  const checkModel = async () => {
    try {
      const response = await axios.get('/api/predict/check-model')
      setModelExists(response.data.exists)
    } catch (error) {
      setModelExists(false)
    }
  }

  useEffect(() => {
    checkModel()
    const interval = setInterval(checkModel, 5000)
    return () => clearInterval(interval)
  }, [])

  const handleCompile = async () => {
    setIsCompiling(true)
    setOutput('Compiling...\n')
    setCompilationStatus(null)
    setGeneratedFiles(null)
    setGeneratedFileContent(null)
    setPredictionResult(null)

    try {
      const response = await axios.post('/api/compile', { code })
      
      if (response.data.success) {
        setOutput(response.data.output)
        setCompilationStatus('success')
        setGeneratedFiles(response.data.files)
        
        // Automatically load train.py content
        if (response.data.files.train) {
          loadGeneratedFile()
        }
        
        // Check if model exists after compilation
        setTimeout(checkModel, 1000)
      } else {
        setOutput(response.data.error || 'Compilation failed')
        setCompilationStatus('error')
      }
    } catch (error) {
      setOutput(`Error: ${error.response?.data?.error || error.message}`)
      setCompilationStatus('error')
    } finally {
      setIsCompiling(false)
    }
  }

  const loadGeneratedFile = async () => {
    try {
      const response = await axios.get('/api/files/train')
      if (response.data.success) {
        setGeneratedFileContent(response.data.content)
      }
    } catch (error) {
      console.error('Error loading generated file:', error)
    }
  }

  const handleDownload = async (fileType) => {
    try {
      const response = await axios.get(`/api/download/${fileType}`, {
        responseType: 'blob'
      })
      
      const url = window.URL.createObjectURL(new Blob([response.data]))
      const link = document.createElement('a')
      link.href = url
      
      if (fileType === 'train') {
        link.setAttribute('download', 'train.py')
      } else if (fileType === 'venv') {
        link.setAttribute('download', 'venv.zip')
      }
      
      document.body.appendChild(link)
      link.click()
      link.parentNode.removeChild(link)
    } catch (error) {
      setOutput(prev => prev + `\nDownload error: ${error.message}`)
    }
  }

  const handleFileUpload = async (event) => {
    const file = event.target.files[0]
    if (!file) return

    setIsPredicting(true)
    setPredictionResult(null)

    try {
      const formData = new FormData()
      formData.append('file', file)

      const response = await axios.post('/api/predict/from-csv', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      })

      if (response.data.success) {
        setPredictionResult(response.data)
      } else {
        setPredictionResult({ error: response.data.error })
      }
    } catch (error) {
      setPredictionResult({ 
        error: error.response?.data?.error || error.message 
      })
    } finally {
      setIsPredicting(false)
    }
  }

  const handleManualPredict = async () => {
    if (!manualInput.trim()) return

    setIsPredicting(true)
    setPredictionResult(null)

    try {
      // Parse comma-separated values
      const features = manualInput.split(',').map(v => parseFloat(v.trim()))
      
      if (features.some(isNaN)) {
        setPredictionResult({ error: 'Invalid input. Please enter comma-separated numbers.' })
        setIsPredicting(false)
        return
      }

      const response = await axios.post('/api/predict/from-input', { features })

      if (response.data.success) {
        setPredictionResult(response.data)
      } else {
        setPredictionResult({ error: response.data.error })
      }
    } catch (error) {
      setPredictionResult({ 
        error: error.response?.data?.error || error.message 
      })
    } finally {
      setIsPredicting(false)
    }
  }

  const examples = [
    {
      name: 'Scikit-Learn',
      code: `dataset "/home/madhava/datasets/classification.csv"

model RandomForestClassifier {
    n_estimators = 100
    max_depth = 4
}`
    },
    {
      name: 'TensorFlow',
      code: DEFAULT_CODE
    },
    {
      name: 'PyTorch',
      code: `dataset "/home/madhava/datasets/images"

model UNet {
    epochs = 20
    batch_size = 16
    learning_rate = 0.0001
}`
    },
    {
      name: 'Transformers',
      code: `dataset "imdb"

model BERT {
    epochs = 3
    batch_size = 8
    learning_rate = 0.00002
}`
    }
  ]

  return (
    <div className="h-screen flex flex-col bg-gray-900 text-white">
      {/* Header */}
      <header className="bg-gray-800 border-b border-gray-700 px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <FileCode className="w-8 h-8 text-blue-500" />
            <div>
              <h1 className="text-2xl font-bold">MLC Compiler IDE</h1>
              <p className="text-sm text-gray-400">Machine Learning Configuration Compiler</p>
            </div>
          </div>
          
          <div className="flex gap-2">
            {examples.map((example) => (
              <button
                key={example.name}
                onClick={() => setCode(example.code)}
                className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg text-sm transition-colors"
              >
                {example.name}
              </button>
            ))}
            <button
              onClick={() => setShowPredictPanel(!showPredictPanel)}
              disabled={!modelExists}
              className={`px-4 py-2 rounded-lg text-sm transition-colors flex items-center gap-2 ${
                modelExists 
                  ? 'bg-green-600 hover:bg-green-700' 
                  : 'bg-gray-600 cursor-not-allowed'
              }`}
            >
              <FlaskConical className="w-4 h-4" />
              {showPredictPanel ? 'Hide Test' : 'Test Model'}
            </button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Editor Panel */}
        <div className="flex-1 flex flex-col border-r border-gray-700" style={{ width: '40%' }}>
          <div className="bg-gray-800 px-4 py-2 border-b border-gray-700 flex items-center justify-between">
            <span className="text-sm font-semibold">config.mlc</span>
            <button
              onClick={handleCompile}
              disabled={isCompiling}
              className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed rounded-lg transition-colors font-medium"
            >
              {isCompiling ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Compiling...
                </>
              ) : (
                <>
                  <PlayCircle className="w-4 h-4" />
                  Compile
                </>
              )}
            </button>
          </div>
          
          <div className="flex-1">
            <Editor
              height="100%"
              defaultLanguage="plaintext"
              theme="vs-dark"
              value={code}
              onChange={(value) => setCode(value || '')}
              options={{
                minimap: { enabled: false },
                fontSize: 14,
                lineNumbers: 'on',
                scrollBeyondLastLine: false,
                automaticLayout: true,
                tabSize: 4,
                wordWrap: 'on'
              }}
            />
          </div>
        </div>

        {/* Generated File Explorer Panel */}
        {generatedFileContent && (
          <div className="flex flex-col bg-gray-850 border-r border-gray-700" style={{ width: '35%' }}>
            <div className="bg-gray-800 px-4 py-2 border-b border-gray-700">
              <div className="flex items-center gap-2 text-sm">
                <Folder className="w-4 h-4 text-blue-400" />
                <span className="font-semibold">Generated Files</span>
              </div>
            </div>
            
            <div className="px-2 py-2 bg-gray-800 border-b border-gray-700">
              <div className="flex items-center gap-2 px-2 py-1 hover:bg-gray-700 rounded cursor-pointer">
                <FileText className="w-4 h-4 text-green-400" />
                <span className="text-sm">train.py</span>
              </div>
            </div>
            
            <div className="flex-1 overflow-hidden">
              <Editor
                height="100%"
                defaultLanguage="python"
                theme="vs-dark"
                value={generatedFileContent}
                options={{
                  readOnly: true,
                  minimap: { enabled: false },
                  fontSize: 13,
                  lineNumbers: 'on',
                  scrollBeyondLastLine: false,
                  automaticLayout: true,
                  tabSize: 4,
                  wordWrap: 'on'
                }}
              />
            </div>
            
            <div className="border-t border-gray-700 p-3 bg-gray-800">
              <button
                onClick={() => handleDownload('train')}
                className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-green-600 hover:bg-green-700 rounded-lg transition-colors text-sm font-medium"
              >
                <Download className="w-4 h-4" />
                Download train.py
              </button>
            </div>
          </div>
        )}

        {/* Output Panel */}
        <div className="flex flex-col bg-gray-850" style={{ width: generatedFileContent ? '25%' : '60%' }}>
          <div className="bg-gray-800 px-4 py-2 border-b border-gray-700 flex items-center gap-2">
            <Terminal className="w-4 h-4" />
            <span className="text-sm font-semibold">Output</span>
            {compilationStatus === 'success' && (
              <CheckCircle className="w-4 h-4 text-green-500 ml-auto" />
            )}
            {compilationStatus === 'error' && (
              <XCircle className="w-4 h-4 text-red-500 ml-auto" />
            )}
          </div>
          
          <div className="flex-1 overflow-auto p-4">
            <pre className="text-sm font-mono text-gray-300 whitespace-pre-wrap">{output || 'Click "Compile" to generate training script...'}</pre>
          </div>
        </div>
      </div>

      {/* Prediction Panel */}
      {showPredictPanel && (
        <div className="absolute right-0 top-16 bottom-12 w-96 bg-gray-800 border-l border-gray-700 shadow-2xl z-10 flex flex-col">
          <div className="bg-gray-900 px-4 py-3 border-b border-gray-700 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <FlaskConical className="w-5 h-5 text-green-500" />
              <h3 className="font-semibold">Test Model</h3>
            </div>
            <button
              onClick={() => setShowPredictPanel(false)}
              className="text-gray-400 hover:text-white"
            >
              ✕
            </button>
          </div>

          <div className="p-4 border-b border-gray-700">
            <div className="flex gap-2 mb-4">
              <button
                onClick={() => setPredictionMode('csv')}
                className={`flex-1 px-3 py-2 rounded text-sm transition-colors ${
                  predictionMode === 'csv'
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                }`}
              >
                <Upload className="w-4 h-4 inline mr-1" />
                Upload CSV
              </button>
              <button
                onClick={() => setPredictionMode('manual')}
                className={`flex-1 px-3 py-2 rounded text-sm transition-colors ${
                  predictionMode === 'manual'
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                }`}
              >
                <Send className="w-4 h-4 inline mr-1" />
                Manual Input
              </button>
            </div>

            {predictionMode === 'csv' ? (
              <div>
                <label className="block text-sm text-gray-400 mb-2">
                  Upload test dataset (CSV format)
                </label>
                <input
                  type="file"
                  accept=".csv"
                  onChange={handleFileUpload}
                  disabled={isPredicting}
                  className="block w-full text-sm text-gray-400 file:mr-4 file:py-2 file:px-4 file:rounded file:border-0 file:text-sm file:font-semibold file:bg-blue-600 file:text-white hover:file:bg-blue-700 file:cursor-pointer cursor-pointer"
                />
              </div>
            ) : (
              <div>
                <label className="block text-sm text-gray-400 mb-2">
                  Enter features (comma-separated)
                </label>
                <input
                  type="text"
                  value={manualInput}
                  onChange={(e) => setManualInput(e.target.value)}
                  placeholder="e.g., 5.1, 3.5, 1.4, 0.2"
                  className="w-full px-3 py-2 bg-gray-700 text-white rounded border border-gray-600 focus:border-blue-500 focus:outline-none text-sm"
                  onKeyPress={(e) => e.key === 'Enter' && handleManualPredict()}
                />
                <button
                  onClick={handleManualPredict}
                  disabled={isPredicting || !manualInput.trim()}
                  className="mt-2 w-full flex items-center justify-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed rounded transition-colors text-sm font-medium"
                >
                  {isPredicting ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin" />
                      Predicting...
                    </>
                  ) : (
                    <>
                      <Send className="w-4 h-4" />
                      Predict
                    </>
                  )}
                </button>
              </div>
            )}
          </div>

          <div className="flex-1 overflow-auto p-4">
            {isPredicting && predictionMode === 'csv' && (
              <div className="flex items-center justify-center h-full">
                <Loader2 className="w-8 h-8 animate-spin text-blue-500" />
              </div>
            )}

            {predictionResult && (
              <div className="space-y-4">
                {predictionResult.error ? (
                  <div className="bg-red-900/30 border border-red-700 rounded p-3">
                    <p className="text-red-300 text-sm">{predictionResult.error}</p>
                  </div>
                ) : (
                  <>
                    <div className="bg-gray-900 rounded p-3 space-y-2">
                      <div className="flex justify-between text-sm">
                        <span className="text-gray-400">Model:</span>
                        <span className="text-white font-mono">{predictionResult.model_type}</span>
                      </div>
                      {predictionResult.num_samples && (
                        <>
                          <div className="flex justify-between text-sm">
                            <span className="text-gray-400">Samples:</span>
                            <span className="text-white">{predictionResult.num_samples}</span>
                          </div>
                          <div className="flex justify-between text-sm">
                            <span className="text-gray-400">Features:</span>
                            <span className="text-white">{predictionResult.num_features}</span>
                          </div>
                        </>
                      )}
                      {predictionResult.accuracy !== null && predictionResult.accuracy !== undefined && (
                        <div className="flex justify-between text-sm pt-2 border-t border-gray-700">
                          <span className="text-gray-400">Accuracy:</span>
                          <span className="text-green-400 font-bold">
                            {(predictionResult.accuracy * 100).toFixed(2)}%
                          </span>
                        </div>
                      )}
                    </div>

                    {predictionResult.prediction !== undefined && (
                      <div className="bg-green-900/30 border border-green-700 rounded p-3">
                        <p className="text-sm text-gray-400 mb-1">Prediction:</p>
                        <p className="text-lg text-green-300 font-bold">{predictionResult.prediction}</p>
                        {predictionResult.probability && (
                          <div className="mt-2 text-xs text-gray-400">
                            Confidence: {Math.max(...predictionResult.probability).toFixed(4)}
                          </div>
                        )}
                      </div>
                    )}

                    {predictionResult.predictions && predictionResult.predictions.length > 0 && (
                      <div className="bg-gray-900 rounded p-3">
                        <p className="text-sm text-gray-400 mb-2">Predictions (first 10):</p>
                        <div className="space-y-1 max-h-60 overflow-y-auto">
                          {predictionResult.predictions.slice(0, 10).map((pred, idx) => (
                            <div key={idx} className="flex justify-between text-xs font-mono">
                              <span className="text-gray-500">Sample {idx + 1}:</span>
                              <div className="flex gap-2">
                                <span className="text-blue-400">Pred: {pred}</span>
                                {predictionResult.actual_labels && (
                                  <span className="text-green-400">
                                    Actual: {predictionResult.actual_labels[idx]}
                                  </span>
                                )}
                              </div>
                            </div>
                          ))}
                        </div>
                        {predictionResult.predictions.length > 10 && (
                          <p className="text-xs text-gray-500 mt-2">
                            ... and {predictionResult.predictions.length - 10} more
                          </p>
                        )}
                      </div>
                    )}
                  </>
                )}
              </div>
            )}

            {!predictionResult && !isPredicting && (
              <div className="text-center text-gray-500 text-sm mt-8">
                <FlaskConical className="w-12 h-12 mx-auto mb-2 opacity-50" />
                <p>Upload a CSV file or enter features manually to test the model</p>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Footer */}
      <footer className="bg-gray-800 border-t border-gray-700 px-6 py-3 text-center text-sm text-gray-400">
        <p>Supports: Scikit-Learn, TensorFlow, PyTorch, Transformers | Auto-detects framework from model name
        {modelExists && <span className="ml-4 text-green-400">● Model Ready</span>}
        </p>
      </footer>
    </div>
  )
}

export default App
