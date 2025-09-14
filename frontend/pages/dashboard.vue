<template>
  <div class="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
    <!-- Header -->
    <div class="bg-black/20 backdrop-blur-sm border-b border-purple-500/20">
      <div class="container mx-auto px-6 py-4">
        <div class="flex items-center justify-between">
          <div class="flex items-center space-x-3">
            <div class="w-8 h-8 bg-gradient-to-r from-purple-500 to-pink-500 rounded-lg flex items-center justify-center">
              <svg class="w-5 h-5 text-white" fill="currentColor" viewBox="0 0 20 20">
                <path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/>
              </svg>
            </div>
            <h1 class="text-2xl font-bold bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
              tempoRoll
            </h1>
          </div>
          <div class="flex items-center space-x-4">
            <div class="flex items-center space-x-2">
              <div :class="connectionStatus === 'connected' ? 'bg-green-500' : 'bg-red-500'" 
                   class="w-2 h-2 rounded-full animate-pulse"></div>
              <span class="text-gray-300 text-sm">{{ connectionStatus }}</span>
            </div>
            <div class="text-gray-300 text-sm">
              Samples: {{ sampleCount }}
            </div>
          </div>
        </div>
      </div>
    </div>

    <div class="container mx-auto px-6 py-8">
      <!-- Current Emotion and Data Row -->
      <div class="grid grid-cols-1 lg:grid-cols-4 gap-6 mb-8">
        <!-- Current Emotion Display -->
        <div class="bg-black/30 backdrop-blur-sm rounded-2xl border border-purple-500/20 p-6 flex items-center justify-center">
          <div class="text-center">
            <div class="relative mb-4">
              <div class="w-20 h-20 mx-auto rounded-full bg-gradient-to-r from-purple-500/20 to-pink-500/20 flex items-center justify-center border-4 border-purple-500/30">
                <span class="text-3xl">{{ getEmotionEmoji(currentEmotion) }}</span>
              </div>
              <div class="absolute -inset-1 rounded-full border-2 border-purple-500/20 animate-pulse"></div>
            </div>
            
            <h3 class="text-xl font-bold text-white mb-1 capitalize">
              {{ currentEmotion || 'Detecting...' }}
            </h3>
            
            <div class="text-gray-400 text-xs">
              {{ lastUpdateTime }}
            </div>
          </div>
        </div>

        <!-- Brainwave Data Numbers -->
        <div class="lg:col-span-3">
          <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div v-for="(value, key) in currentBrainwaveData" :key="key" 
                 class="bg-black/30 backdrop-blur-sm rounded-xl border border-purple-500/20 p-4">
              <div class="text-gray-400 text-xs uppercase tracking-wider mb-1">{{ getDisplayName(key) }}</div>
              <div class="text-white text-lg font-mono mb-2">{{ getDisplayValue(key, value) }}</div>
              <div class="w-full bg-gray-700 rounded-full h-2">
                <div class="bg-gradient-to-r from-purple-500 to-pink-500 h-2 rounded-full transition-all duration-300"
                     :style="{ width: getProgressWidth(key, value) + '%' }"></div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Wave Visualizations Full Width Row -->
      <div class="bg-black/30 backdrop-blur-sm rounded-2xl border border-purple-500/20 p-6">
        <h2 class="text-xl font-semibold text-white mb-6 flex items-center">
          <svg class="w-6 h-6 mr-2 text-purple-400" fill="currentColor" viewBox="0 0 20 20">
            <path d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"/>
          </svg>
          Live Brainwave Patterns
        </h2>
        
        <!-- Wave Visualizations - 2x3 Grid -->
        <div class="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
          <!-- Beta Waves -->
          <div class="bg-black/20 rounded-lg p-4">
            <h4 class="text-sm font-medium text-purple-300 mb-2">Beta Waves (Focus & Activity)</h4>
            <div class="text-xs text-gray-400 mb-2">Low Beta (Purple) • High Beta (Pink)</div>
            <canvas ref="betaCanvas" class="w-full h-32"></canvas>
          </div>
          
          <!-- Alpha Waves -->
          <div class="bg-black/20 rounded-lg p-4">
            <h4 class="text-sm font-medium text-blue-300 mb-2">Alpha Waves (Relaxation & Awareness)</h4>
            <div class="text-xs text-gray-400 mb-2">Low Alpha (Cyan) • High Alpha (Green)</div>
            <canvas ref="alphaCanvas" class="w-full h-32"></canvas>
          </div>
          
          <!-- Gamma Waves -->
          <div class="bg-black/20 rounded-lg p-4">
            <h4 class="text-sm font-medium text-pink-300 mb-2">Gamma Waves (High Cognition)</h4>
            <div class="text-xs text-gray-400 mb-2">Low Gamma (Magenta) • Mid Gamma (Hot Pink)</div>
            <canvas ref="gammaCanvas" class="w-full h-32"></canvas>
          </div>
          
          <!-- Delta & Theta Waves -->
          <div class="bg-black/20 rounded-lg p-4">
            <h4 class="text-sm font-medium text-amber-300 mb-2">Deep Brain Waves</h4>
            <div class="text-xs text-gray-400 mb-2">Delta (Amber) • Theta (Red)</div>
            <canvas ref="deltaThetaCanvas" class="w-full h-32"></canvas>
          </div>
          
          <!-- Mental States -->
          <div class="bg-black/20 rounded-lg p-4">
            <h4 class="text-sm font-medium text-indigo-300 mb-2">Mental States</h4>
            <div class="text-xs text-gray-400 mb-2">Attention (Indigo) • Meditation (Lime)</div>
            <canvas ref="mentalCanvas" class="w-full h-32"></canvas>
          </div>
        </div>
      </div>

      <!-- Emotion History -->
      <div class="bg-black/30 backdrop-blur-sm rounded-2xl border border-purple-500/20 p-6 mt-8">
        <h2 class="text-xl font-semibold text-white mb-6 flex items-center">
          <svg class="w-6 h-6 mr-2 text-purple-400" fill="currentColor" viewBox="0 0 20 20">
            <path d="M3 4a1 1 0 011-1h12a1 1 0 011 1v2a1 1 0 01-1 1H4a1 1 0 01-1-1V4zM3 10a1 1 0 011-1h6a1 1 0 011 1v6a1 1 0 01-1 1H4a1 1 0 01-1-1v-6zM14 9a1 1 0 00-1 1v6a1 1 0 001 1h2a1 1 0 001-1v-6a1 1 0 00-1-1h-2z"/>
          </svg>
          Emotion Timeline
        </h2>
        
        <div class="space-y-3 max-h-64 overflow-y-auto">
          <div v-for="(emotion, index) in emotionHistory" :key="index"
               class="flex items-center justify-between bg-black/20 rounded-lg p-4 border border-gray-700/50">
            <div class="flex items-center space-x-3">
              <span class="text-2xl">{{ getEmotionEmoji(emotion.emotion) }}</span>
              <div>
                <div class="text-white font-medium capitalize">{{ emotion.emotion }}</div>
                <div class="text-gray-400 text-sm">{{ formatTime(emotion.timestamp) }}</div>
              </div>
            </div>
            <div class="text-gray-400 text-sm">
              {{ emotion.sample_count }} samples
            </div>
          </div>
          
          <div v-if="emotionHistory.length === 0" class="text-center text-gray-500 py-8">
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
const sampleCount = ref(0)
const lastUpdateTime = ref('')
const currentBrainwaveData = ref({})
const emotionHistory = ref([])
const brainwaveDataPoints = ref([])

// SSE connection
let eventSource = null
let betaCanvas = ref(null)
let alphaCanvas = ref(null) 
let gammaCanvas = ref(null)
let deltaThetaCanvas = ref(null)
let mentalCanvas = ref(null)
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
      sampleCount.value = data.sample_count
      
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
  }
  
  // Initialize Alpha canvas  
  if (alphaCanvas.value) {
    alphaContext = alphaCanvas.value.getContext('2d')
    alphaCanvas.value.width = alphaCanvas.value.offsetWidth
    alphaCanvas.value.height = alphaCanvas.value.offsetHeight
  }
  
  // Initialize Gamma canvas
  if (gammaCanvas.value) {
    gammaContext = gammaCanvas.value.getContext('2d')
    gammaCanvas.value.width = gammaCanvas.value.offsetWidth
    gammaCanvas.value.height = gammaCanvas.value.offsetHeight
  }
  
  // Initialize Delta/Theta canvas
  if (deltaThetaCanvas.value) {
    deltaThetaContext = deltaThetaCanvas.value.getContext('2d')
    deltaThetaCanvas.value.width = deltaThetaCanvas.value.offsetWidth
    deltaThetaCanvas.value.height = deltaThetaCanvas.value.offsetHeight
  }
  
  // Initialize Mental States canvas
  if (mentalCanvas.value) {
    mentalContext = mentalCanvas.value.getContext('2d')
    mentalCanvas.value.width = mentalCanvas.value.offsetWidth
    mentalCanvas.value.height = mentalCanvas.value.offsetHeight
  }
}

function updateWaveVisualization() {
  if (brainwaveDataPoints.value.length === 0) return
  
  // Update Beta Waves (lowBeta, highBeta)
  if (betaContext) {
    const canvas = betaCanvas.value
    const ctx = betaContext
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    
    const betaKeys = ['lowBeta', 'highBeta']
    const betaColors = ['#8b5cf6', '#ec4899'] // Purple, Pink
    
    betaKeys.forEach((key, keyIndex) => {
      ctx.beginPath()
      ctx.strokeStyle = betaColors[keyIndex]
      ctx.lineWidth = 2
      
      brainwaveDataPoints.value.forEach((dataPoint, index) => {
        const x = (index / brainwaveDataPoints.value.length) * canvas.width
        const value = dataPoint[key] || 0
        // Normalize beta values (they're typically higher, so scale down)
        const normalizedValue = Math.min(value / 50000, 1) // Scale for typical beta range
        const y = canvas.height - (normalizedValue * canvas.height)
        
        if (index === 0) {
          ctx.moveTo(x, y)
        } else {
          ctx.lineTo(x, y)
        }
      })
      
      ctx.stroke()
    })
  }
  
  // Update Alpha Waves (lowAlpha, highAlpha)
  if (alphaContext) {
    const canvas = alphaCanvas.value
    const ctx = alphaContext
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    
    const alphaKeys = ['lowAlpha', 'highAlpha']
    const alphaColors = ['#06b6d4', '#10b981'] // Cyan, Green
    
    alphaKeys.forEach((key, keyIndex) => {
      ctx.beginPath()
      ctx.strokeStyle = alphaColors[keyIndex]
      ctx.lineWidth = 2
      
      brainwaveDataPoints.value.forEach((dataPoint, index) => {
        const x = (index / brainwaveDataPoints.value.length) * canvas.width
        const value = dataPoint[key] || 0
        // Normalize alpha values
        const normalizedValue = Math.min(value / 50000, 1) // Scale for typical alpha range
        const y = canvas.height - (normalizedValue * canvas.height)
        
        if (index === 0) {
          ctx.moveTo(x, y)
        } else {
          ctx.lineTo(x, y)
        }
      })
      
      ctx.stroke()
    })
  }
  
  // Update Gamma Waves (lowGamma, midGamma)
  if (gammaContext) {
    const canvas = gammaCanvas.value
    const ctx = gammaContext
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    
    const gammaKeys = ['lowGamma', 'midGamma']
    const gammaColors = ['#ec4899', '#be185d'] // Hot Pink, Dark Pink
    
    gammaKeys.forEach((key, keyIndex) => {
      ctx.beginPath()
      ctx.strokeStyle = gammaColors[keyIndex]
      ctx.lineWidth = 2
      
      brainwaveDataPoints.value.forEach((dataPoint, index) => {
        const x = (index / brainwaveDataPoints.value.length) * canvas.width
        const value = dataPoint[key] || 0
        // Normalize gamma values (typically lower than other bands)
        const normalizedValue = Math.min(value / 20000, 1) // Scale for typical gamma range
        const y = canvas.height - (normalizedValue * canvas.height)
        
        if (index === 0) {
          ctx.moveTo(x, y)
        } else {
          ctx.lineTo(x, y)
        }
      })
      
      ctx.stroke()
    })
  }
  
  // Update Delta/Theta Waves
  if (deltaThetaContext) {
    const canvas = deltaThetaCanvas.value
    const ctx = deltaThetaContext
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    
    const deltaThetaKeys = ['delta', 'theta']
    const deltaThetaColors = ['#f59e0b', '#ef4444'] // Amber, Red
    
    deltaThetaKeys.forEach((key, keyIndex) => {
      ctx.beginPath()
      ctx.strokeStyle = deltaThetaColors[keyIndex]
      ctx.lineWidth = 2
      
      brainwaveDataPoints.value.forEach((dataPoint, index) => {
        const x = (index / brainwaveDataPoints.value.length) * canvas.width
        const value = dataPoint[key] || 0
        // Delta and theta are typically very high values
        const normalizedValue = Math.min(value / 200000, 1)
        const y = canvas.height - (normalizedValue * canvas.height)
        
        if (index === 0) {
          ctx.moveTo(x, y)
        } else {
          ctx.lineTo(x, y)
        }
      })
      
      ctx.stroke()
    })
  }
  
  // Update Mental States (attention, meditation)
  if (mentalContext) {
    const canvas = mentalCanvas.value
    const ctx = mentalContext
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    
    const mentalKeys = ['attention', 'meditation']
    const mentalColors = ['#6366f1', '#84cc16'] // Indigo, Lime
    
    mentalKeys.forEach((key, keyIndex) => {
      ctx.beginPath()
      ctx.strokeStyle = mentalColors[keyIndex]
      ctx.lineWidth = 2
      
      brainwaveDataPoints.value.forEach((dataPoint, index) => {
        const x = (index / brainwaveDataPoints.value.length) * canvas.width
        const value = dataPoint[key] || 0
        // Attention and meditation are 0-100 scale
        const normalizedValue = value / 100
        const y = canvas.height - (normalizedValue * canvas.height)
        
        if (index === 0) {
          ctx.moveTo(x, y)
        } else {
          ctx.lineTo(x, y)
        }
      })
      
      ctx.stroke()
    })
  }
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
  background: rgba(0, 0, 0, 0.1);
  border-radius: 3px;
}

::-webkit-scrollbar-thumb {
  background: rgba(139, 92, 246, 0.5);
  border-radius: 3px;
}

::-webkit-scrollbar-thumb:hover {
  background: rgba(139, 92, 246, 0.7);
}
</style>
