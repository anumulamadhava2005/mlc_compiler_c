import { useState, useEffect } from 'react'
import Editor from '@monaco-editor/react'
import { PlayCircle, Download, FileCode, Terminal, Loader2, CheckCircle, XCircle, FileText, Folder } from 'lucide-react'
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

  const handleCompile = async () => {
    setIsCompiling(true)
    setOutput('Compiling...\n')
    setCompilationStatus(null)
    setGeneratedFiles(null)
    setGeneratedFileContent(null)

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

      {/* Footer */}
      <footer className="bg-gray-800 border-t border-gray-700 px-6 py-3 text-center text-sm text-gray-400">
        <p>Supports: Scikit-Learn, TensorFlow, PyTorch, Transformers | Auto-detects framework from model name</p>
      </footer>
    </div>
  )
}

export default App
