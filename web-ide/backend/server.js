import express from 'express'
import cors from 'cors'
import { exec } from 'child_process'
import { promisify } from 'util'
import fs from 'fs/promises'
import path from 'path'
import { fileURLToPath } from 'url'
import archiver from 'archiver'
import multer from 'multer'
import axios from 'axios'

const execAsync = promisify(exec)
const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

const app = express()
const PORT = 5000

// Paths
const COMPILER_DIR = path.join(__dirname, '..', '..')
const COMPILER_PATH = path.join(COMPILER_DIR, 'mlc_compiler')
const TEMP_DIR = path.join(__dirname, 'temp')
const TEMP_MLC_FILE = path.join(TEMP_DIR, 'temp.mlc')

// Middleware
app.use(cors())
app.use(express.json())

// Ensure temp directory exists
await fs.mkdir(TEMP_DIR, { recursive: true })

// Configure multer for file uploads
const storage = multer.diskStorage({
  destination: TEMP_DIR,
  filename: (req, file, cb) => {
    cb(null, `upload_${Date.now()}_${file.originalname}`)
  }
})
const upload = multer({ storage })

// Prediction API URL
const PREDICT_API_URL = 'http://localhost:5001'

// Compile endpoint
app.post('/api/compile', async (req, res) => {
  try {
    const { code } = req.body

    if (!code) {
      return res.status(400).json({ success: false, error: 'No code provided' })
    }

    // Write code to temporary .mlc file
    await fs.writeFile(TEMP_MLC_FILE, code, 'utf-8')

    // Check if compiler exists
    try {
      await fs.access(COMPILER_PATH)
    } catch (error) {
      return res.status(500).json({ 
        success: false, 
        error: 'MLC compiler not found. Please run "make" in the mlc_compiler_c directory first.' 
      })
    }

    // Run the compiler
    let output = ''
    try {
      // Change to compiler directory and run compiler
      const { stdout, stderr } = await execAsync(
        `cd "${COMPILER_DIR}" && "${COMPILER_PATH}" "${TEMP_MLC_FILE}"`,
        { 
          cwd: COMPILER_DIR,
          timeout: 30000 // 30 second timeout
        }
      )
      
      output = stdout + (stderr ? `\nStderr: ${stderr}` : '')

      // Check if train.py was generated
      const trainPyPath = path.join(COMPILER_DIR, 'train.py')
      const venvPath = path.join(COMPILER_DIR, 'venv')

      let trainPyExists = false
      let venvExists = false

      try {
        await fs.access(trainPyPath)
        trainPyExists = true
      } catch {}

      try {
        await fs.access(venvPath)
        const venvStats = await fs.stat(venvPath)
        venvExists = venvStats.isDirectory()
      } catch {}

      res.json({
        success: true,
        output: output,
        files: {
          train: trainPyExists,
          venv: venvExists
        }
      })

    } catch (error) {
      // Compilation error
      const errorOutput = error.stdout || error.stderr || error.message
      res.json({
        success: false,
        error: `Compilation failed:\n${errorOutput}`,
        output: errorOutput
      })
    }

  } catch (error) {
    console.error('Server error:', error)
    res.status(500).json({ 
      success: false, 
      error: `Server error: ${error.message}` 
    })
  }
})

// Get train.py content
app.get('/api/files/train', async (req, res) => {
  try {
    const trainPyPath = path.join(COMPILER_DIR, 'train.py')
    
    try {
      const content = await fs.readFile(trainPyPath, 'utf-8')
      res.json({ success: true, content, filename: 'train.py' })
    } catch {
      return res.status(404).json({ success: false, error: 'train.py not found. Please compile first.' })
    }
  } catch (error) {
    res.status(500).json({ success: false, error: error.message })
  }
})

// Download train.py
app.get('/api/download/train', async (req, res) => {
  try {
    const trainPyPath = path.join(COMPILER_DIR, 'train.py')
    
    try {
      await fs.access(trainPyPath)
    } catch {
      return res.status(404).json({ error: 'train.py not found. Please compile first.' })
    }

    res.download(trainPyPath, 'train.py')
  } catch (error) {
    res.status(500).json({ error: error.message })
  }
})

// Download venv as zip
app.get('/api/download/venv', async (req, res) => {
  try {
    const venvPath = path.join(COMPILER_DIR, 'venv')
    
    try {
      await fs.access(venvPath)
    } catch {
      return res.status(404).json({ error: 'venv directory not found. Please compile first.' })
    }

    res.setHeader('Content-Type', 'application/zip')
    res.setHeader('Content-Disposition', 'attachment; filename=venv.zip')

    const archive = archiver('zip', {
      zlib: { level: 9 }
    })

    archive.on('error', (err) => {
      throw err
    })

    archive.pipe(res)
    archive.directory(venvPath, 'venv')
    await archive.finalize()

  } catch (error) {
    res.status(500).json({ error: error.message })
  }
})

// Check if model exists
app.get('/api/predict/check-model', async (req, res) => {
  try {
    const response = await axios.get(`${PREDICT_API_URL}/api/predict/check-model`)
    res.json(response.data)
  } catch (error) {
    if (error.code === 'ECONNREFUSED') {
      res.status(503).json({ 
        exists: false, 
        error: 'Prediction API not running. Please start predict_api.py' 
      })
    } else {
      res.status(500).json({ exists: false, error: error.message })
    }
  }
})

// Predict from uploaded CSV
app.post('/api/predict/from-csv', upload.single('file'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ success: false, error: 'No file uploaded' })
    }

    // Create form data to send to Flask API
    const FormData = (await import('form-data')).default
    const formData = new FormData()
    const fileStream = await fs.readFile(req.file.path)
    formData.append('file', fileStream, req.file.originalname)

    // Forward to Flask API
    const response = await axios.post(`${PREDICT_API_URL}/api/predict/from-csv`, formData, {
      headers: formData.getHeaders()
    })

    // Clean up uploaded file
    await fs.unlink(req.file.path)

    res.json(response.data)
  } catch (error) {
    if (error.code === 'ECONNREFUSED') {
      res.status(503).json({ 
        success: false, 
        error: 'Prediction API not running. Please start predict_api.py' 
      })
    } else {
      res.status(500).json({ 
        success: false, 
        error: error.response?.data?.error || error.message 
      })
    }
  }
})

// Predict from manual input
app.post('/api/predict/from-input', async (req, res) => {
  try {
    const response = await axios.post(`${PREDICT_API_URL}/api/predict/from-input`, req.body)
    res.json(response.data)
  } catch (error) {
    if (error.code === 'ECONNREFUSED') {
      res.status(503).json({ 
        success: false, 
        error: 'Prediction API not running. Please start predict_api.py' 
      })
    } else {
      res.status(500).json({ 
        success: false, 
        error: error.response?.data?.error || error.message 
      })
    }
  }
})

// Health check
app.get('/api/health', (req, res) => {
  res.json({ status: 'ok' })
})

app.listen(PORT, () => {
  console.log(`🚀 MLC Compiler Backend running on http://localhost:${PORT}`)
  console.log(`📂 Compiler directory: ${COMPILER_DIR}`)
  console.log(`🔧 Compiler path: ${COMPILER_PATH}`)
  console.log(`\n💡 To enable predictions, start the Flask API:`)
  console.log(`   cd backend && python3 predict_api.py`)
})
