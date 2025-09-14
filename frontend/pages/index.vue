<template>
  <div class="min-h-screen bg-gradient-to-br from-blue-50 via-sky-100 to-blue-100"> 
    <!-- Hero Section -->
    <div class="container mx-auto px-6 py-16">
      <div class="text-center">
        <!-- Logo and Title -->
        <div class="flex items-center justify-center space-x-4 mb-8">
          <div class="w-20 h-20 bg-gradient-to-r from-sky-500 to-blue-600 rounded-2xl flex items-center justify-center shadow-lg">
            <svg class="w-12 h-12 text-white" fill="currentColor" viewBox="0 0 20 20">
              <path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/>
            </svg>
          </div>
          <h1 class="text-7xl font-bold bg-gradient-to-r from-blue-600 to-sky-500 bg-clip-text text-transparent">
            tempoRoll
          </h1>
        </div>
        
        <!-- Subtitle -->
        <p class="text-2xl text-slate-700 mb-8 max-w-3xl mx-auto font-medium">
          Revolutionary Real-time Brainwave Emotion Detection & Therapeutic Music Generation
        </p>
        
        <p class="text-lg text-slate-600 mb-8 max-w-2xl mx-auto">
          Monitor your mental state with advanced AI-powered analysis, get instant emotional insights, and receive personalized therapeutic music recommendations to optimize your wellbeing.
        </p>
        
        <!-- Mental Health Applications -->
        <div class="bg-white/50 backdrop-blur-sm rounded-2xl border border-blue-200 shadow-lg p-8 mb-20 max-w-4xl mx-auto">
          <div class="text-center mb-6">
            <div class="w-16 h-16 bg-gradient-to-r from-green-100 to-emerald-100 rounded-xl flex items-center justify-center mx-auto mb-4">
              <svg class="w-8 h-8 text-green-600" fill="currentColor" viewBox="0 0 20 20">
                <path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/>
              </svg>
            </div>
            <h3 class="text-2xl font-bold text-slate-800 mb-4">Clinical Mental Health Support</h3>
          </div>
          
          <p class="text-lg text-slate-700 leading-relaxed text-center max-w-3xl mx-auto">
            Our advanced brainwave analysis system can help <strong>counteract and manage symptoms</strong> of various mental health conditions including <strong>schizophrenic episodes</strong>, <strong>bipolar disorders</strong>, anxiety, depression, and ADHD. By detecting early warning signs through real-time EEG monitoring, the system provides immediate therapeutic music interventions designed to stabilize mood, reduce agitation, and promote emotional regulation before symptoms escalate.
          </p>
          
          <div class="grid md:grid-cols-3 gap-6 mt-8">
            <div class="text-center">
              <div class="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center mx-auto mb-3">
                <svg class="w-6 h-6 text-blue-600" fill="currentColor" viewBox="0 0 20 20">
                  <path d="M13 6a3 3 0 11-6 0 3 3 0 016 0zM18 8a2 2 0 11-4 0 2 2 0 014 0zM14 15a4 4 0 00-8 0v3h8v-3z"/>
                </svg>
              </div>
              <h4 class="font-semibold text-slate-800 mb-2">Early Detection</h4>
              <p class="text-slate-600 text-sm">Identify mood changes and episode onset before they fully manifest</p>
            </div>
            
            <div class="text-center">
              <div class="w-12 h-12 bg-purple-100 rounded-lg flex items-center justify-center mx-auto mb-3">
                <svg class="w-6 h-6 text-purple-600" fill="currentColor" viewBox="0 0 20 20">
                  <path d="M18 3a1 1 0 00-1.196-.98l-10 2A1 1 0 006 5v6.114A4.369 4.369 0 005 11a4 4 0 104 4V5.114l8-1.6V9.114A4.369 4.369 0 0016 9a4 4 0 104 4V3z"/>
                </svg>
              </div>
              <h4 class="font-semibold text-slate-800 mb-2">Therapeutic Intervention</h4>
              <p class="text-slate-600 text-sm">Immediate personalized music therapy to stabilize emotional states</p>
            </div>
            
            <div class="text-center">
              <div class="w-12 h-12 bg-green-100 rounded-lg flex items-center justify-center mx-auto mb-3">
                <svg class="w-6 h-6 text-green-600" fill="currentColor" viewBox="0 0 20 20">
                  <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-8.293l-3-3a1 1 0 00-1.414 0l-3 3a1 1 0 001.414 1.414L9 9.414V13a1 1 0 102 0V9.414l1.293 1.293a1 1 0 001.414-1.414z" clip-rule="evenodd"/>
                </svg>
              </div>
              <h4 class="font-semibold text-slate-800 mb-2">Symptom Management</h4>
              <p class="text-slate-600 text-sm">Continuous monitoring and support for long-term mental health stability</p>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- How It Works Section -->
    <div class="bg-white/40 backdrop-blur-sm py-20">
      <div class="container mx-auto px-6">
        <div class="text-center mb-16">
          <h2 class="text-4xl font-bold text-slate-800 mb-4">How It Works</h2>
          <p class="text-xl text-slate-600 max-w-3xl mx-auto">
            Our cutting-edge system combines neuroscience, AI, and music therapy to provide real-time mental health insights
          </p>
        </div>

        <!-- User with EEG Device Photo -->
        <div class="grid lg:grid-cols-2 gap-16 items-center mb-20">
          <div class="order-2 lg:order-1">
            <div class="bg-white/60 backdrop-blur-sm rounded-3xl border border-blue-200 shadow-lg p-2 mb-8">
              <img src="/assets/images/jennifer.JPG" 
                   alt="User with EEG Device" 
                   class="w-full h-96 object-cover rounded-2xl"/>
            </div>
            
            <!-- Try It Out Button -->
            <div class="text-center">
              <NuxtLink to="/dashboard" 
                        class="inline-flex items-center px-8 py-4 bg-gradient-to-r from-purple-500 to-pink-500 text-white font-semibold text-lg rounded-2xl hover:from-purple-600 hover:to-pink-600 transition-all duration-300 transform hover:scale-105 shadow-lg">
                <svg class="w-5 h-5 mr-2" fill="currentColor" viewBox="0 0 20 20">
                  <path d="M10 12a2 2 0 100-4 2 2 0 000 4z"/>
                  <path fill-rule="evenodd" d="M.458 10C1.732 5.943 5.522 3 10 3s8.268 2.943 9.542 7c-1.274 4.057-5.064 7-9.542 7S1.732 14.057.458 10zM14 10a4 4 0 11-8 0 4 4 0 018 0z" clip-rule="evenodd"/>
                </svg>
                Try It Out
              </NuxtLink>
              <p class="text-slate-500 text-sm mt-3">Experience real-time brainwave analysis</p>
            </div>
          </div>
          
          <div class="order-1 lg:order-2">
            <h3 class="text-3xl font-bold text-slate-800 mb-6">Step 1: EEG Data Capture</h3>
            <p class="text-lg text-slate-600 mb-6">
              Our system connects to professional EEG devices to capture real-time brainwave signals across multiple frequency bands including Alpha, Beta, Theta, Delta, and Gamma waves.
            </p>
            <div class="space-y-4">
              <div class="flex items-center space-x-3">
                <div class="w-8 h-8 bg-green-100 rounded-full flex items-center justify-center">
                  <svg class="w-4 h-4 text-green-600" fill="currentColor" viewBox="0 0 20 20">
                    <path fill-rule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clip-rule="evenodd"/>
                  </svg>
                </div>
                <span class="text-slate-700">10 channels of brainwave data</span>
              </div>
              <div class="flex items-center space-x-3">
                <div class="w-8 h-8 bg-green-100 rounded-full flex items-center justify-center">
                  <svg class="w-4 h-4 text-green-600" fill="currentColor" viewBox="0 0 20 20">
                    <path fill-rule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clip-rule="evenodd"/>
                  </svg>
                </div>
                <span class="text-slate-700">Real-time signal processing</span>
              </div>
              <div class="flex items-center space-x-3">
                <div class="w-8 h-8 bg-green-100 rounded-full flex items-center justify-center">
                  <svg class="w-4 h-4 text-green-600" fill="currentColor" viewBox="0 0 20 20">
                    <path fill-rule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clip-rule="evenodd"/>
                  </svg>
                </div>
                <span class="text-slate-700">Advanced noise filtering</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Features Section -->
    <div class="container mx-auto px-6 py-20">
      <div class="text-center mb-16">
        <h2 class="text-4xl font-bold text-slate-800 mb-4">Powerful Features</h2>
        <p class="text-xl text-slate-600 max-w-3xl mx-auto">
          Advanced AI-powered analysis with beautiful visualizations and therapeutic recommendations
        </p>
      </div>

      <!-- Features Grid -->
      <div class="grid md:grid-cols-2 lg:grid-cols-4 gap-8 mb-16">
        <div class="bg-white/60 backdrop-blur-sm rounded-2xl border border-blue-200 shadow-lg p-8 text-center">
          <div class="w-16 h-16 bg-purple-100 rounded-xl flex items-center justify-center mb-6 mx-auto">
            <svg class="w-8 h-8 text-purple-600" fill="currentColor" viewBox="0 0 20 20">
              <path d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"/>
            </svg>
          </div>
          <h3 class="text-xl font-semibold text-slate-800 mb-3">Live Brainwave Analysis</h3>
          <p class="text-slate-600">Real-time processing of multi-channel EEG data with advanced signal filtering and frequency analysis</p>
        </div>
        
        <div class="bg-white/60 backdrop-blur-sm rounded-2xl border border-blue-200 shadow-lg p-8 text-center">
          <div class="w-16 h-16 bg-blue-100 rounded-xl flex items-center justify-center mb-6 mx-auto">
            <svg class="w-8 h-8 text-blue-600" fill="currentColor" viewBox="0 0 20 20">
              <path d="M10 2L3 7v11a1 1 0 001 1h3v-5h6v5h3a1 1 0 001-1V7l-7-5z"/>
            </svg>
          </div>
          <h3 class="text-xl font-semibold text-slate-800 mb-3">AI Emotion Detection</h3>
          <p class="text-slate-600">Machine learning models trained on neural networks for accurate emotion classification and mental state analysis</p>
        </div>
        
        <div class="bg-white/60 backdrop-blur-sm rounded-2xl border border-blue-200 shadow-lg p-8 text-center">
          <div class="w-16 h-16 bg-sky-100 rounded-xl flex items-center justify-center mb-6 mx-auto">
            <svg class="w-8 h-8 text-sky-600" fill="currentColor" viewBox="0 0 20 20">
              <path d="M3 4a1 1 0 011-1h12a1 1 0 011 1v2a1 1 0 01-1 1H4a1 1 0 01-1-1V4zM3 10a1 1 0 011-1h6a1 1 0 011 1v6a1 1 0 01-1 1H4a1 1 0 01-1-1v-6zM14 9a1 1 0 00-1 1v6a1 1 0 001 1h2a1 1 0 001-1v-6a1 1 0 00-1-1h-2z"/>
            </svg>
          </div>
          <h3 class="text-xl font-semibold text-slate-800 mb-3">Beautiful Visualizations</h3>
          <p class="text-slate-600">Interactive charts and real-time wave visualization with comprehensive history tracking and analytics</p>
        </div>

        <div class="bg-white/60 backdrop-blur-sm rounded-2xl border border-blue-200 shadow-lg p-8 text-center">
          <div class="w-16 h-16 bg-pink-100 rounded-xl flex items-center justify-center mb-6 mx-auto">
            <svg class="w-8 h-8 text-pink-600" fill="currentColor" viewBox="0 0 20 20">
              <path d="M18 3a1 1 0 00-1.196-.98l-10 2A1 1 0 006 5v6.114A4.369 4.369 0 005 11a4 4 0 104 4V5.114l8-1.6V9.114A4.369 4.369 0 0016 9a4 4 0 104 4V3z"/>
            </svg>
          </div>
          <h3 class="text-xl font-semibold text-slate-800 mb-3">Therapeutic Music</h3>
          <p class="text-slate-600">AI-generated personalized music recommendations designed to counteract negative emotions and promote wellbeing</p>
        </div>
      </div>
    </div>

    <!-- App Screenshots Section -->
    <div class="bg-white/40 backdrop-blur-sm py-20">
      <div class="container mx-auto px-6">
        <div class="text-center mb-16">
          <h2 class="text-4xl font-bold text-slate-800 mb-4">See It In Action</h2>
          <p class="text-xl text-slate-600 max-w-3xl mx-auto">
            Experience the power of real-time brainwave analysis with our intuitive dashboard
          </p>
        </div>

        <!-- App Screenshots - Left Right Left Layout -->
        <div class="space-y-16">
          <!-- First Screenshot - Left -->
          <div class="grid lg:grid-cols-2 gap-12 items-center">
            <div class="order-1 lg:order-1">
              <div class="bg-white/60 backdrop-blur-sm rounded-2xl border border-blue-200 shadow-lg p-4 transform hover:scale-105 transition-all duration-500 hover:shadow-2xl animate-pulse hover:animate-none">
                <img src="/assets/images/dashboard.png" 
                     alt="Real-time Dashboard Screenshot" 
                     class="w-full h-80 object-cover rounded-xl transition-transform duration-300 hover:scale-105"/>
              </div>
            </div>
            <div class="order-2 lg:order-2">
              <div class="transform hover:translate-x-2 transition-transform duration-300">
                <h4 class="text-3xl font-bold text-slate-800 mb-4">Real-time Dashboard and Detailed Analysis</h4>
                <p class="text-lg text-slate-600 leading-relaxed">
                  Experience live emotion detection with our comprehensive dashboard. Monitor your mental state in real-time with advanced AI-powered analysis, brainwave visualization, and personalized therapeutic music recommendations.
                </p>
                <div class="mt-6 space-y-3">
                  <div class="flex items-center space-x-3">
                    <div class="w-2 h-2 bg-blue-500 rounded-full animate-bounce"></div>
                    <span class="text-slate-700">Live emotion classification</span>
                  </div>
                  <div class="flex items-center space-x-3">
                    <div class="w-2 h-2 bg-purple-500 rounded-full animate-bounce" style="animation-delay: 0.1s"></div>
                    <span class="text-slate-700">Real-time brainwave data</span>
                  </div>
                  <div class="flex items-center space-x-3">
                    <div class="w-2 h-2 bg-pink-500 rounded-full animate-bounce" style="animation-delay: 0.2s"></div>
                    <span class="text-slate-700">Therapeutic recommendations</span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <!-- Second Screenshot - Right -->
          <div class="grid lg:grid-cols-2 gap-12 items-center">
            <div class="order-2 lg:order-1">
              <div class="transform hover:-translate-x-2 transition-transform duration-300">
                <h4 class="text-3xl font-bold text-slate-800 mb-4">Live Wave Analysis</h4>
                <p class="text-lg text-slate-600 leading-relaxed">
                  Visualize your brainwaves in real-time with our interactive charts. Track Alpha, Beta, Theta, Delta, and Gamma wave patterns to understand your mental state and cognitive performance.
                </p>
                <div class="mt-6 space-y-3">
                  <div class="flex items-center space-x-3">
                    <div class="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                    <span class="text-slate-700">Multi-frequency analysis</span>
                  </div>
                  <div class="flex items-center space-x-3">
                    <div class="w-2 h-2 bg-yellow-500 rounded-full animate-pulse" style="animation-delay: 0.1s"></div>
                    <span class="text-slate-700">Interactive wave charts</span>
                  </div>
                  <div class="flex items-center space-x-3">
                    <div class="w-2 h-2 bg-red-500 rounded-full animate-pulse" style="animation-delay: 0.2s"></div>
                    <span class="text-slate-700">Historical tracking</span>
                  </div>
                </div>
              </div>
            </div>
            <div class="order-1 lg:order-2">
              <div class="bg-white/60 backdrop-blur-sm rounded-2xl border border-blue-200 shadow-lg p-4 transform hover:scale-105 transition-all duration-500 hover:shadow-2xl animate-pulse hover:animate-none">
                <img src="/assets/images/waves.png" 
                     alt="Live Brainwave Charts Screenshot" 
                     class="w-full h-80 object-cover rounded-xl transition-transform duration-300 hover:scale-105"/>
              </div>
            </div>
          </div>

          <!-- Third Screenshot - Left -->
          <div class="grid lg:grid-cols-2 gap-12 items-center">
            <div class="order-1 lg:order-1">
              <div class="bg-white/60 backdrop-blur-sm rounded-2xl border border-blue-200 shadow-lg p-4 transform hover:scale-105 transition-all duration-500 hover:shadow-2xl">
                <div class="bg-gradient-to-br from-slate-100 to-slate-200 rounded-xl h-80 flex items-center justify-center transition-all duration-300 hover:from-blue-50 hover:to-sky-50">
                  <div class="text-center transform hover:scale-110 transition-transform duration-300">
                    <svg class="w-20 h-20 text-slate-400 mx-auto mb-4 animate-spin" style="animation-duration: 3s" fill="currentColor" viewBox="0 0 20 20">
                      <path fill-rule="evenodd" d="M3 5a2 2 0 012-2h10a2 2 0 012 2v8a2 2 0 01-2 2h-2.22l.123.489.804.804A1 1 0 0113 18H7a1 1 0 01-.707-1.707l.804-.804L7.22 15H5a2 2 0 01-2-2V5zm5.771 7H5V5h10v7H8.771z" clip-rule="evenodd"/>
                    </svg>
                    <p class="text-slate-500 font-medium text-lg">Comprehensive Analysis</p>
                    <p class="text-slate-400">Coming Soon</p>
                  </div>
                </div>
              </div>
            </div>
            <div class="order-2 lg:order-2">
              <div class="transform hover:translate-x-2 transition-transform duration-300">
                <h4 class="text-3xl font-bold text-slate-800 mb-4">Comprehensive Mental Health Insights</h4>
                <p class="text-lg text-slate-600 leading-relaxed">
                  Get detailed analysis of your emotional patterns, stress levels, and cognitive performance. Our AI provides personalized insights to help you optimize your mental wellness journey.
                </p>
                <div class="mt-6 space-y-3">
                  <div class="flex items-center space-x-3">
                    <div class="w-2 h-2 bg-indigo-500 rounded-full animate-bounce"></div>
                    <span class="text-slate-700">Emotional pattern analysis</span>
                  </div>
                  <div class="flex items-center space-x-3">
                    <div class="w-2 h-2 bg-teal-500 rounded-full animate-bounce" style="animation-delay: 0.1s"></div>
                    <span class="text-slate-700">Stress level monitoring</span>
                  </div>
                  <div class="flex items-center space-x-3">
                    <div class="w-2 h-2 bg-orange-500 rounded-full animate-bounce" style="animation-delay: 0.2s"></div>
                    <span class="text-slate-700">Personalized recommendations</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- CTA Section -->
    <div class="container mx-auto px-6 py-20 text-center">
      <div class="bg-white/60 backdrop-blur-sm rounded-3xl border border-blue-200 shadow-lg p-12">
        <h2 class="text-4xl font-bold text-slate-800 mb-6">Ready to Transform Your Mental Wellness?</h2>
        <p class="text-xl text-slate-600 mb-8 max-w-2xl mx-auto">
          Start your journey with real-time brainwave analysis and personalized therapeutic recommendations
        </p>
        
        <NuxtLink to="/dashboard" 
                  class="inline-flex items-center px-12 py-6 bg-gradient-to-r from-blue-500 to-sky-500 text-white font-semibold text-xl rounded-2xl hover:from-blue-600 hover:to-sky-600 transition-all duration-300 transform hover:scale-105 shadow-xl">
          <svg class="w-6 h-6 mr-3" fill="currentColor" viewBox="0 0 20 20">
            <path d="M10 2L3 7v11a1 1 0 001 1h3v-5h6v5h3a1 1 0 001-1V7l-7-5z"/>
          </svg>
          Get Started Now
        </NuxtLink>
        
        <div class="text-slate-600 text-sm mt-6">
          Compatible with professional EEG devices • Real-time processing • AI-powered insights
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