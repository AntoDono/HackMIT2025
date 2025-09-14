<template>
  <div class="min-h-screen bg-gradient-to-br from-blue-50 via-sky-100 to-blue-100">
    <!-- Header -->
    <div class="bg-white/40 backdrop-blur-sm border-b border-blue-200 shadow-sm">
      <div class="container mx-auto px-6 py-4">
        <div class="flex items-center justify-between">
          <div class="flex items-center space-x-3">
            <div class="w-8 h-8 bg-gradient-to-r from-sky-500 to-blue-600 rounded-lg flex items-center justify-center">
              <svg class="w-5 h-5 text-white" fill="currentColor" viewBox="0 0 20 20">
                <path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/>
              </svg>
            </div>
            <h1 class="text-2xl font-bold bg-gradient-to-r from-blue-600 to-sky-500 bg-clip-text text-transparent">
              tempoRoll
            </h1>
          </div>
          <div class="flex items-center space-x-4">
            <div class="flex items-center space-x-2">
              <div :class="connectionStatus === 'connected' ? 'bg-green-500' : 'bg-red-500'" 
                   class="w-2 h-2 rounded-full animate-pulse"></div>
              <span class="text-slate-600 text-sm">{{ connectionStatus }}</span>
            </div>
            <div class="text-slate-600 text-sm">
              Samples: {{ sampleCount }}
            </div>
          </div>
        </div>
      </div>
    </div>

    <div class="container mx-auto px-6 py-8">
      <!-- Current Emotion and Data Row -->
      <div class="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
        <!-- Current Emotion Display -->
        <div class="lg:col-span-1 bg-white/60 backdrop-blur-sm rounded-2xl border border-blue-200 shadow-lg p-6">
          <!-- Emotion Status Grid -->
          <div class="grid grid-cols-2 gap-4 mb-4">
            <!-- Estimated Emotion (ML Classification) -->
            <div class="text-center">
              <div class="relative mb-3">
                <div class="w-12 h-12 mx-auto rounded-full bg-gradient-to-r from-blue-100 to-sky-100 flex items-center justify-center border-3 border-blue-200">
                  <span class="text-lg">{{ getEmotionEmoji(currentEmotion) }}</span>
                </div>
                <div class="absolute -inset-1 rounded-full border-2 border-blue-300 animate-pulse"></div>
              </div>
              <div class="text-xs text-slate-500 uppercase tracking-wider mb-1">
                Estimated Emotion
              </div>
              <h3 class="text-sm font-bold text-slate-800 mb-1 capitalize">
                {{ currentEmotion || 'Detecting...' }}
              </h3>
              <div class="text-xs text-slate-400">
                Real-time ML classification
              </div>
            </div>
            
            <!-- Current Status (LLM Analysis) -->
            <div class="text-center">
              <div class="relative mb-3">
                <div class="w-12 h-12 mx-auto rounded-full bg-gradient-to-r from-purple-100 to-pink-100 flex items-center justify-center border-3 border-purple-200">
                  <span class="text-lg">{{ getEmotionEmoji(currentStatus) }}</span>
                </div>
                <div class="absolute -inset-1 rounded-full border-2 border-purple-300 animate-pulse"></div>
              </div>
              <div class="text-xs text-slate-500 uppercase tracking-wider mb-1">
                Current Status
              </div>
              <h3 class="text-sm font-bold text-slate-800 mb-1 capitalize">
                {{ currentStatus || 'Analyzing...' }}
              </h3>
              <div class="text-xs text-slate-400">
                Most accurate analysis
              </div>
            </div>
          </div>
          
          <div class="text-slate-500 text-xs text-center mb-4">
            {{ lastUpdateTime }}
          </div>
          
          <!-- Real-time Emotion Analysis -->
          <div class="border-t border-slate-200 pt-3">
            <div class="text-xs text-slate-500 mb-1 font-bold">Detailed Analysis:</div>
            <div class="text-sm text-slate-700 leading-relaxed">
              {{ currentEmotionAnalysis || 'Analyzing brainwave patterns...' }}
            </div>
          </div>
        </div>

        <!-- Brainwave Data Numbers -->
        <div class="lg:col-span-2">
          <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div v-for="(value, key) in currentBrainwaveData" :key="key" 
                 class="bg-white/60 backdrop-blur-sm rounded-xl border border-blue-200 shadow-lg p-4">
              <div class="text-slate-500 text-xs uppercase tracking-wider mb-1">{{ getDisplayName(key) }}</div>
              <div class="text-slate-800 text-lg font-mono mb-2">{{ getDisplayValue(key, value) }}</div>
              <div class="w-full bg-slate-200 rounded-full h-2">
                <div class="bg-gradient-to-r from-blue-500 to-sky-400 h-2 rounded-full transition-all duration-300"
                     :style="{ width: getProgressWidth(key, value) + '%' }"></div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Counteract Audio Stimuli -->
      <div class="bg-white/60 backdrop-blur-sm rounded-2xl border border-blue-200 shadow-lg p-6 mb-8">
        <h2 class="text-xl font-semibold text-slate-800 mb-6 flex items-center">
          <svg class="w-6 h-6 mr-2 text-purple-600" fill="currentColor" viewBox="0 0 20 20">
            <path d="M18 3a1 1 0 00-1.196-.98l-10 2A1 1 0 006 5v6.114A4.369 4.369 0 005 11a4 4 0 104 4V5.114l8-1.6V9.114A4.369 4.369 0 0016 9a4 4 0 104 4V3z"/>
          </svg>
          Counteract Audio Stimuli
        </h2>
        
        <div class="bg-white/40 rounded-lg p-4 border border-blue-100">
          <div class="text-sm text-slate-700 leading-relaxed">
            {{ songReasoning }}
          </div>
        </div>
      </div>

      <!-- Wave Visualizations Full Width Row -->
      <div class="bg-white/60 backdrop-blur-sm rounded-2xl border border-blue-200 shadow-lg p-6">
        <h2 class="text-xl font-semibold text-slate-800 mb-6 flex items-center">
          <svg class="w-6 h-6 mr-2 text-purple-600" fill="currentColor" viewBox="0 0 20 20">
            <path d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"/>
          </svg>
          Live Brainwave Patterns
        </h2>
        
        <!-- Wave Visualizations - 2 per row Grid -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <!-- Beta Waves -->
          <div class="bg-white/40 rounded-lg p-4 border border-blue-100">
            <h4 class="text-sm font-medium text-purple-600 mb-2">Beta Waves (Focus & Activity)</h4>
            <div class="text-xs text-slate-500 mb-2">Low Beta (Purple) • High Beta (Pink)</div>
            <div class="relative">
              <canvas ref="betaCanvas" class="w-full h-40 cursor-crosshair"></canvas>
              <div ref="betaTooltip" class="absolute bg-black/80 text-white text-xs px-2 py-1 rounded pointer-events-none opacity-0 transition-opacity z-10"></div>
            </div>
          </div>
          
          <!-- Alpha Waves -->
          <div class="bg-white/40 rounded-lg p-4 border border-blue-100">
            <h4 class="text-sm font-medium text-blue-600 mb-2">Alpha Waves (Relaxation & Awareness)</h4>
            <div class="text-xs text-slate-500 mb-2">Low Alpha (Cyan) • High Alpha (Green)</div>
            <div class="relative">
              <canvas ref="alphaCanvas" class="w-full h-40 cursor-crosshair"></canvas>
              <div ref="alphaTooltip" class="absolute bg-black/80 text-white text-xs px-2 py-1 rounded pointer-events-none opacity-0 transition-opacity z-10"></div>
            </div>
          </div>
          
          <!-- Gamma Waves -->
          <div class="bg-white/40 rounded-lg p-4 border border-blue-100">
            <h4 class="text-sm font-medium text-pink-600 mb-2">Gamma Waves (High Cognition)</h4>
            <div class="text-xs text-slate-500 mb-2">Low Gamma (Magenta) • Mid Gamma (Hot Pink)</div>
            <div class="relative">
              <canvas ref="gammaCanvas" class="w-full h-40 cursor-crosshair"></canvas>
              <div ref="gammaTooltip" class="absolute bg-black/80 text-white text-xs px-2 py-1 rounded pointer-events-none opacity-0 transition-opacity z-10"></div>
            </div>
          </div>
          
          <!-- Delta & Theta Waves -->
          <div class="bg-white/40 rounded-lg p-4 border border-blue-100">
            <h4 class="text-sm font-medium text-amber-600 mb-2">Deep Brain Waves</h4>
            <div class="text-xs text-slate-500 mb-2">Delta (Amber) • Theta (Red)</div>
            <div class="relative">
              <canvas ref="deltaThetaCanvas" class="w-full h-40 cursor-crosshair"></canvas>
              <div ref="deltaThetaTooltip" class="absolute bg-black/80 text-white text-xs px-2 py-1 rounded pointer-events-none opacity-0 transition-opacity z-10"></div>
            </div>
          </div>
          
          <!-- Mental States -->
          <div class="bg-white/40 rounded-lg p-4 border border-blue-100">
            <h4 class="text-sm font-medium text-indigo-600 mb-2">Mental States</h4>
            <div class="text-xs text-slate-500 mb-2">Attention (Indigo) • Meditation (Lime)</div>
            <div class="relative">
              <canvas ref="mentalCanvas" class="w-full h-40 cursor-crosshair"></canvas>
              <div ref="mentalTooltip" class="absolute bg-black/80 text-white text-xs px-2 py-1 rounded pointer-events-none opacity-0 transition-opacity z-10"></div>
            </div>
          </div>
        </div>
      </div>

      <!-- Emotion History -->
      <div class="bg-white/60 backdrop-blur-sm rounded-2xl border border-blue-200 shadow-lg p-6 mt-8">
        <h2 class="text-xl font-semibold text-slate-800 mb-6 flex items-center">
          <svg class="w-6 h-6 mr-2 text-purple-600" fill="currentColor" viewBox="0 0 20 20">
            <path d="M3 4a1 1 0 011-1h12a1 1 0 011 1v2a1 1 0 01-1 1H4a1 1 0 01-1-1V4zM3 10a1 1 0 011-1h6a1 1 0 011 1v6a1 1 0 01-1 1H4a1 1 0 01-1-1v-6zM14 9a1 1 0 00-1 1v6a1 1 0 001 1h2a1 1 0 001-1v-6a1 1 0 00-1-1h-2z"/>
          </svg>
          Emotion Timeline
        </h2>
        
        <div class="space-y-3 max-h-64 overflow-y-auto">
          <div v-for="(emotion, index) in emotionHistory" :key="index"
               class="flex items-center justify-between bg-white/40 rounded-lg p-4 border border-blue-100">
            <div class="flex items-center space-x-3">
              <span class="text-2xl">{{ getEmotionEmoji(emotion.emotion) }}</span>
              <div>
                <div class="text-slate-800 font-medium capitalize">{{ emotion.emotion }}</div>
                <div class="text-slate-500 text-sm">{{ formatTime(emotion.timestamp) }}</div>
              </div>
            </div>
            <div class="text-slate-500 text-sm">
              {{ emotion.sample_count }} samples
            </div>
          </div>
          
          <div v-if="emotionHistory.length === 0" class="text-center text-slate-500 py-8">
            No emotions detected yet. Start sending brainwave data!
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue'

// Reactive data
const connectionStatus = ref('disconnected')
const currentEmotion = ref('')
const currentEmotionAnalysis = ref('')
const currentStatus = ref('unknown')
const songReasoning = ref('No music recommendation yet')
const sampleCount = ref(0)
const lastUpdateTime = ref('')
const currentBrainwaveData = ref({})
const currentLabeledData = ref({})
const emotionHistory = ref([])
const brainwaveDataPoints = ref([])

// SSE connection
let eventSource = null
let betaCanvas = ref(null)
let alphaCanvas = ref(null) 
let gammaCanvas = ref(null)
let deltaThetaCanvas = ref(null)
let mentalCanvas = ref(null)
let betaTooltip = ref(null)
let alphaTooltip = ref(null)
let gammaTooltip = ref(null)
let deltaThetaTooltip = ref(null)
let mentalTooltip = ref(null)
let betaContext = null
let alphaContext = null
let gammaContext = null
let deltaThetaContext = null
let mentalContext = null

// Emotion emoji mapping
const emotionEmojis = {
  'focused': '🎯',
  'relaxed': '😌',
  'relax': '😌',
  'stressed': '😰',
  'stress': '😰',
  'happy': '😊',
  'sad': '😢',
  'excited': '🤩',
  'nervous': '😬',
  'calm': '😇',
  'surprise': '😲',
  'angry': '😠'
}

function getEmotionEmoji(emotion) {
  return emotionEmojis[emotion?.toLowerCase()] || '🧠'
}

function formatTime(timestamp) {
  return new Date(timestamp).toLocaleTimeString()
}

function getDisplayName(key) {
  const nameMap = {
    'lowAlpha': 'Low Alpha',
    'highAlpha': 'High Alpha', 
    'lowBeta': 'Low Beta',
    'highBeta': 'High Beta',
    'lowGamma': 'Low Gamma',
    'midGamma': 'Mid Gamma',
    'delta': 'Delta',
    'theta': 'Theta',
    'attention': 'Attention',
    'meditation': 'Meditation'
  }
  return nameMap[key] || key
}

function getDisplayValue(key, value) {
  if (!value) return '0.00'
  
  switch(key) {
    case 'attention':
    case 'meditation':
      // These are 0-100 scale, show as percentage
      return `${value.toFixed(0)}%`
      
    case 'delta':
    case 'theta':
    case 'lowAlpha':
    case 'highAlpha':
    case 'lowBeta':
    case 'highBeta':
    case 'lowGamma':
    case 'midGamma':
      // Display microvolts squared with proper Greek mu symbol
      if (value >= 1000000) {
        return `${(value / 1000000).toFixed(1)}M μV²`
      } else if (value >= 1000) {
        return `${(value / 1000).toFixed(1)}k μV²`
      } else {
        return `${value.toFixed(1)} μV²`
      }
      
    default:
      return value.toFixed(2)
  }
}

function getProgressWidth(key, value) {
  if (!value) return 0
  
  // Different scaling for different types of brainwave data
  switch(key) {
    case 'attention':
    case 'meditation':
      // These are 0-100 scale
      return Math.min(value, 100)
      
    case 'delta':
    case 'theta':
      // These are very high values, scale them down
      return Math.min((value / 200000) * 100, 100)
      
    case 'lowAlpha':
    case 'highAlpha':
    case 'lowBeta':
    case 'highBeta':
      // These are medium-high values
      return Math.min((value / 50000) * 100, 100)
      
    case 'lowGamma':
    case 'midGamma':
      // Gamma waves are typically lower values
      return Math.min((value / 20000) * 100, 100)
      
    default:
      return Math.min(value / 100, 100)
  }
}

// Calculate dynamic max value for auto-adjusting bounds
function calculateDynamicMax(dataPoints, key, defaultMax) {
  if (dataPoints.length === 0) return defaultMax
  
  const values = dataPoints.map(point => point[key] || 0).filter(val => val > 0)
  if (values.length === 0) return defaultMax
  
  const maxValue = Math.max(...values)
  const minValue = Math.min(...values)
  
  // Add 20% padding above the max value for better visualization
  const paddedMax = maxValue * 1.2
  
  // Ensure we don't go below a reasonable minimum
  const reasonableMin = defaultMax * 0.1
  
  return Math.max(paddedMax, reasonableMin)
}

// Draw grid and axes
function drawGridAndAxes(ctx, canvas, timeRange = '2min') {
  ctx.save()
  ctx.strokeStyle = 'rgba(148, 163, 184, 0.2)'
  ctx.lineWidth = 1
  
  // Draw horizontal grid lines
  for (let i = 0; i <= 4; i++) {
    const y = (i / 4) * canvas.height
    ctx.beginPath()
    ctx.moveTo(0, y)
    ctx.lineTo(canvas.width, y)
    ctx.stroke()
  }
  
  // Draw vertical grid lines
  for (let i = 0; i <= 4; i++) {
    const x = (i / 4) * canvas.width
    ctx.beginPath()
    ctx.moveTo(x, 0)
    ctx.lineTo(x, canvas.height)
    ctx.stroke()
  }
  
  // Draw axes labels
  ctx.fillStyle = 'rgba(100, 116, 139, 0.8)'
  ctx.font = '10px system-ui'
  ctx.textAlign = 'right'
  ctx.fillText('High', canvas.width - 2, 12)
  ctx.fillText('Low', canvas.width - 2, canvas.height - 2)
  ctx.textAlign = 'left'
  ctx.fillText('Now', 2, canvas.height - 2)
  ctx.fillText(`-${timeRange}`, 2, 12)
  
  ctx.restore()
}

// Setup mouse events for tooltip
function setupCanvasTooltip(canvas, tooltip, keys, colors, defaultMaxValues) {
  if (!canvas || !tooltip) return
  
  canvas.addEventListener('mousemove', (event) => {
    const rect = canvas.getBoundingClientRect()
    const x = event.clientX - rect.left
    const y = event.clientY - rect.top
    
    // Calculate data point index
    const dataIndex = Math.round((x / canvas.width) * (brainwaveDataPoints.value.length - 1))
    
    if (dataIndex >= 0 && dataIndex < brainwaveDataPoints.value.length) {
      const dataPoint = brainwaveDataPoints.value[dataIndex]
      let tooltipContent = ''
      
      keys.forEach((key, index) => {
        const value = dataPoint[key] || 0
        const displayValue = getDisplayValue(key, value)
        const dynamicMax = calculateDynamicMax(brainwaveDataPoints.value, key, defaultMaxValues[index])
        const percentage = ((value / dynamicMax) * 100).toFixed(1)
        tooltipContent += `<span style="color: ${colors[index]}">${getDisplayName(key)}: ${displayValue} (${percentage}%)</span><br>`
      })
      
      tooltip.innerHTML = tooltipContent
      tooltip.style.left = `${x + 10}px`
      tooltip.style.top = `${y - 10}px`
      tooltip.style.opacity = '1'
    }
  })
  
  canvas.addEventListener('mouseleave', () => {
    tooltip.style.opacity = '0'
  })
}

function connectSSE() {
  try {
    eventSource = new EventSource('http://localhost:8001')
    
    eventSource.onopen = () => {
      connectionStatus.value = 'connected'
      console.log('SSE connected')
    }
    
    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        handleSSEMessage(data)
      } catch (error) {
        console.error('Error parsing SSE message:', error)
      }
    }
    
    eventSource.onerror = (error) => {
      console.error('SSE error:', error)
      connectionStatus.value = 'disconnected'
      
      // Attempt to reconnect after 3 seconds
      setTimeout(() => {
        if (eventSource) {
          eventSource.close()
        }
        connectSSE()
      }, 3000)
    }
  } catch (error) {
    console.error('Failed to connect SSE:', error)
    connectionStatus.value = 'error'
  }
}

function handleSSEMessage(data) {
  switch (data.type) {
    case 'initial_data':
      emotionHistory.value = data.emotion_history || []
      sampleCount.value = data.sample_count || 0
      break
      
    case 'emotion_update':
      currentEmotion.value = data.current_emotion
      sampleCount.value = data.sample_count
      lastUpdateTime.value = new Date().toLocaleTimeString()
      currentBrainwaveData.value = data.brainwave_data || {}
      
      // Update emotion history
      if (data.emotion_history) {
        emotionHistory.value = data.emotion_history
      }
      break
      
    case 'brainwave_data':
      currentBrainwaveData.value = data.data || {}
      currentLabeledData.value = data.labeled_data || {}
      currentEmotionAnalysis.value = data.current_emotion_analysis || ''
      currentStatus.value = data.current_status || 'unknown'
      songReasoning.value = data.song_reasoning || 'No music recommendation yet'
      sampleCount.value = data.sample_count
      lastUpdateTime.value = new Date().toLocaleTimeString()
      
      // Add to wave visualization data
      brainwaveDataPoints.value.push(data.data)
      if (brainwaveDataPoints.value.length > 100) {
        brainwaveDataPoints.value.shift()
      }
      
      // Update wave visualization
      updateWaveVisualization()
      break
  }
}

function initCanvas() {
  // Initialize Beta canvas
  if (betaCanvas.value) {
    betaContext = betaCanvas.value.getContext('2d')
    betaCanvas.value.width = betaCanvas.value.offsetWidth
    betaCanvas.value.height = betaCanvas.value.offsetHeight
    setupCanvasTooltip(betaCanvas.value, betaTooltip.value, ['lowBeta', 'highBeta'], ['#8b5cf6', '#ec4899'], [50000, 50000])
  }
  
  // Initialize Alpha canvas  
  if (alphaCanvas.value) {
    alphaContext = alphaCanvas.value.getContext('2d')
    alphaCanvas.value.width = alphaCanvas.value.offsetWidth
    alphaCanvas.value.height = alphaCanvas.value.offsetHeight
    setupCanvasTooltip(alphaCanvas.value, alphaTooltip.value, ['lowAlpha', 'highAlpha'], ['#06b6d4', '#10b981'], [50000, 50000])
  }
  
  // Initialize Gamma canvas
  if (gammaCanvas.value) {
    gammaContext = gammaCanvas.value.getContext('2d')
    gammaCanvas.value.width = gammaCanvas.value.offsetWidth
    gammaCanvas.value.height = gammaCanvas.value.offsetHeight
    setupCanvasTooltip(gammaCanvas.value, gammaTooltip.value, ['lowGamma', 'midGamma'], ['#ec4899', '#be185d'], [20000, 20000])
  }
  
  // Initialize Delta/Theta canvas
  if (deltaThetaCanvas.value) {
    deltaThetaContext = deltaThetaCanvas.value.getContext('2d')
    deltaThetaCanvas.value.width = deltaThetaCanvas.value.offsetWidth
    deltaThetaCanvas.value.height = deltaThetaCanvas.value.offsetHeight
    setupCanvasTooltip(deltaThetaCanvas.value, deltaThetaTooltip.value, ['delta', 'theta'], ['#f59e0b', '#ef4444'], [200000, 200000])
  }
  
  // Initialize Mental States canvas
  if (mentalCanvas.value) {
    mentalContext = mentalCanvas.value.getContext('2d')
    mentalCanvas.value.width = mentalCanvas.value.offsetWidth
    mentalCanvas.value.height = mentalCanvas.value.offsetHeight
    setupCanvasTooltip(mentalCanvas.value, mentalTooltip.value, ['attention', 'meditation'], ['#6366f1', '#84cc16'], [100, 100])
  }
}

function drawWaveChart(context, canvas, keys, colors, defaultMaxValues) {
  if (!context || !canvas) return
  
  const ctx = context
  ctx.clearRect(0, 0, canvas.width, canvas.height)
  
  // Draw grid and axes first
  drawGridAndAxes(ctx, canvas)
  
  keys.forEach((key, keyIndex) => {
    // Calculate dynamic max value based on actual data
    const dynamicMaxValue = calculateDynamicMax(brainwaveDataPoints.value, key, defaultMaxValues[keyIndex])
    const color = colors[keyIndex]
    
    // Draw main wave line
    ctx.beginPath()
    ctx.strokeStyle = color
    ctx.lineWidth = 2.5
    ctx.lineCap = 'round'
    ctx.lineJoin = 'round'
    
    brainwaveDataPoints.value.forEach((dataPoint, index) => {
      const x = (index / (brainwaveDataPoints.value.length - 1)) * canvas.width
      const value = dataPoint[key] || 0
      const normalizedValue = Math.min(value / dynamicMaxValue, 1)
      const y = canvas.height - (normalizedValue * canvas.height)
      
      if (index === 0) {
        ctx.moveTo(x, y)
      } else {
        ctx.lineTo(x, y)
      }
    })
    ctx.stroke()
    
    // Draw data points for better visibility
    if (brainwaveDataPoints.value.length < 50) {
      ctx.fillStyle = color
      brainwaveDataPoints.value.forEach((dataPoint, index) => {
        const x = (index / (brainwaveDataPoints.value.length - 1)) * canvas.width
        const value = dataPoint[key] || 0
        const normalizedValue = Math.min(value / dynamicMaxValue, 1)
        const y = canvas.height - (normalizedValue * canvas.height)
        
        ctx.beginPath()
        ctx.arc(x, y, 2, 0, 2 * Math.PI)
        ctx.fill()
      })
    }
  })
}

function updateWaveVisualization() {
  if (brainwaveDataPoints.value.length === 0) return
  
  // Update Beta Waves
  drawWaveChart(betaContext, betaCanvas.value, ['lowBeta', 'highBeta'], ['#8b5cf6', '#ec4899'], [50000, 50000])
  
  // Update Alpha Waves
  drawWaveChart(alphaContext, alphaCanvas.value, ['lowAlpha', 'highAlpha'], ['#06b6d4', '#10b981'], [50000, 50000])
  
  // Update Gamma Waves
  drawWaveChart(gammaContext, gammaCanvas.value, ['lowGamma', 'midGamma'], ['#ec4899', '#be185d'], [20000, 20000])
  
  // Update Delta/Theta Waves
  drawWaveChart(deltaThetaContext, deltaThetaCanvas.value, ['delta', 'theta'], ['#f59e0b', '#ef4444'], [200000, 200000])
  
  // Update Mental States
  drawWaveChart(mentalContext, mentalCanvas.value, ['attention', 'meditation'], ['#6366f1', '#84cc16'], [100, 100])
}

onMounted(() => {
  connectSSE()
  initCanvas()
  
  // Handle window resize
  window.addEventListener('resize', initCanvas)
})

onUnmounted(() => {
  if (eventSource) {
    eventSource.close()
  }
  window.removeEventListener('resize', initCanvas)
})
</script>

<style scoped>
/* Custom scrollbar */
::-webkit-scrollbar {
  width: 6px;
}

::-webkit-scrollbar-track {
  background: rgba(148, 163, 184, 0.2);
  border-radius: 3px;
}

::-webkit-scrollbar-thumb {
  background: rgba(59, 130, 246, 0.5);
  border-radius: 3px;
}

::-webkit-scrollbar-thumb:hover {
  background: rgba(59, 130, 246, 0.7);
}
</style>
