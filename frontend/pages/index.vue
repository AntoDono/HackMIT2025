<template>
  <div class="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
    <div class="container mx-auto px-6 py-16">
      <div class="text-center">
        <!-- Logo and Title -->
        <div class="flex items-center justify-center space-x-4 mb-8">
          <div class="w-16 h-16 bg-gradient-to-r from-purple-500 to-pink-500 rounded-2xl flex items-center justify-center">
            <svg class="w-10 h-10 text-white" fill="currentColor" viewBox="0 0 20 20">
              <path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/>
            </svg>
          </div>
          <h1 class="text-6xl font-bold bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
            tempoRoll
          </h1>
        </div>
        
        <!-- Subtitle -->
        <p class="text-xl text-gray-300 mb-12 max-w-2xl mx-auto">
          Real-time brainwave emotion detection and visualization. 
          Monitor your mental state with advanced AI-powered analysis.
        </p>
        
        <!-- Features Grid -->
        <div class="grid md:grid-cols-3 gap-8 mb-16">
          <div class="bg-black/30 backdrop-blur-sm rounded-2xl border border-purple-500/20 p-8">
            <div class="w-12 h-12 bg-purple-500/20 rounded-lg flex items-center justify-center mb-4 mx-auto">
              <svg class="w-6 h-6 text-purple-400" fill="currentColor" viewBox="0 0 20 20">
                <path d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"/>
              </svg>
            </div>
            <h3 class="text-xl font-semibold text-white mb-2">Live Brainwave Analysis</h3>
            <p class="text-gray-400">Real-time processing of 8-channel EEG data with advanced signal filtering</p>
          </div>
          
          <div class="bg-black/30 backdrop-blur-sm rounded-2xl border border-purple-500/20 p-8">
            <div class="w-12 h-12 bg-pink-500/20 rounded-lg flex items-center justify-center mb-4 mx-auto">
              <svg class="w-6 h-6 text-pink-400" fill="currentColor" viewBox="0 0 20 20">
                <path d="M10 2L3 7v11a1 1 0 001 1h3v-5h6v5h3a1 1 0 001-1V7l-7-5z"/>
              </svg>
            </div>
            <h3 class="text-xl font-semibold text-white mb-2">Emotion Detection</h3>
            <p class="text-gray-400">AI-powered emotion classification trained on neural network models</p>
          </div>
          
          <div class="bg-black/30 backdrop-blur-sm rounded-2xl border border-purple-500/20 p-8">
            <div class="w-12 h-12 bg-cyan-500/20 rounded-lg flex items-center justify-center mb-4 mx-auto">
              <svg class="w-6 h-6 text-cyan-400" fill="currentColor" viewBox="0 0 20 20">
                <path d="M3 4a1 1 0 011-1h12a1 1 0 011 1v2a1 1 0 01-1 1H4a1 1 0 01-1-1V4zM3 10a1 1 0 011-1h6a1 1 0 011 1v6a1 1 0 01-1 1H4a1 1 0 01-1-1v-6zM14 9a1 1 0 00-1 1v6a1 1 0 001 1h2a1 1 0 001-1v-6a1 1 0 00-1-1h-2z"/>
              </svg>
            </div>
            <h3 class="text-xl font-semibold text-white mb-2">Beautiful Visualization</h3>
            <p class="text-gray-400">Interactive charts and real-time wave visualization with history tracking</p>
          </div>
        </div>
        
        <!-- CTA Button -->
        <div class="space-y-4">
          <NuxtLink to="/dashboard" 
                    class="inline-flex items-center px-8 py-4 bg-gradient-to-r from-purple-500 to-pink-500 text-white font-semibold rounded-2xl hover:from-purple-600 hover:to-pink-600 transition-all duration-300 transform hover:scale-105 shadow-lg">
            <svg class="w-5 h-5 mr-2" fill="currentColor" viewBox="0 0 20 20">
              <path d="M10 2L3 7v11a1 1 0 001 1h3v-5h6v5h3a1 1 0 001-1V7l-7-5z"/>
            </svg>
            Launch Dashboard
          </NuxtLink>
          
          <div class="text-gray-400 text-sm">
            Make sure your EEG device is connected and the backend server is running
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
// Set page title
useHead({
  title: 'tempoRoll - Real-time Brainwave Emotion Detection'
})
</script>