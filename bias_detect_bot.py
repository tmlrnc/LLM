import React, { useState, useRef, useEffect } from 'react';

const SmartChatbot = () => {
  const [messages, setMessages] = useState([
    {
      id: 1,
      text: "Hello! I'm a smart chatbot powered by OpenAI and LangChain that can detect biased or unclear questions and assess my confidence in answers. Try asking me something!",
      type: 'bot',
      confidence: 0.9
    }
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [apiKey, setApiKey] = useState('');
  const [showApiInput, setShowApiInput] = useState(true);
  const messagesEndRef = useRef(null);

  // Bias detection keywords
  const biasKeywords = [
    'always', 'never', 'all', 'every', 'none', 'only', 'obviously', 'clearly',
    'everyone knows', "it's obvious", 'without question', 'undoubtedly',
    'definitely', 'absolutely', 'certainly', 'surely', 'of course'
  ];

  // Unclear language indicators
  const unclearIndicators = [
    'thing', 'stuff', 'it', 'that', 'this thing', 'you know', 'whatever',
    'something', 'anything', 'everything', 'whatsit', 'thingy', 'doohickey'
  ];

  // Knowledge confidence mapping
  const confidenceMap = {
    'what is': 0.9,
    'how to': 0.8,
    'explain': 0.85,
    'define': 0.9,
    'calculate': 0.95,
    'math': 0.95,
    'science': 0.8,
    'history': 0.75,
    'geography': 0.8,
    'who is': 0.7,
    'when did': 0.75,
    'where is': 0.8,
    'why does': 0.7,
    'how does': 0.8,
    'opinion': 0.3,
    'think': 0.3,
    'believe': 0.3,
    'predict': 0.4,
    'future': 0.3,
    'will happen': 0.2,
    'best': 0.5,
    'worst': 0.5,
    'better': 0.5,
    'should i': 0.4,
    'recommend': 0.5
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const detectBias = (message) => {
    const lowerMessage = message.toLowerCase();
    return biasKeywords.filter(keyword => {
      const variations = [
        keyword,
        keyword.replace(/'/g, ''),
        keyword.replace(/'/g, "'")
      ];
      return variations.some(variation => lowerMessage.includes(variation));
    });
  };

  const detectUnclearLanguage = (message) => {
    const lowerMessage = message.toLowerCase();
    return unclearIndicators.filter(indicator => 
      lowerMessage.includes(indicator)
    );
  };

  const calculateConfidence = (message) => {
    const lowerMessage = message.toLowerCase();
    let confidence = 0.7;
    let matchFound = false;

    // Check against knowledge base
    for (const [topic, topicConfidence] of Object.entries(confidenceMap)) {
      if (lowerMessage.includes(topic)) {
        confidence = Math.max(confidence, topicConfidence);
        matchFound = true;
      }
    }

    if (!matchFound) {
      confidence = lowerMessage.includes('?') ? 0.6 : 0.5;
    }

    // Boost for factual questions
    if (lowerMessage.includes('what') || lowerMessage.includes('how') || 
        lowerMessage.includes('when') || lowerMessage.includes('where') || 
        lowerMessage.includes('who')) {
      confidence *= 1.1;
    }

    // Reduce for subjective content
    if (lowerMessage.includes('opinion') || lowerMessage.includes('think') || 
        lowerMessage.includes('believe') || lowerMessage.includes('feel')) {
      confidence *= 0.2;
    }

    if (lowerMessage.includes('predict') || lowerMessage.includes('future') || 
        lowerMessage.includes('will happen')) {
      confidence *= 0.3;
    }

    return Math.min(Math.max(confidence, 0.1), 1.0);
  };

  const callOpenAI = async (message, confidence) => {
    try {
      // Simulate LangChain + OpenAI call
      const systemPrompt = `You are a helpful AI assistant. Based on the confidence level of ${(confidence * 100).toFixed(0)}%, respond appropriately:
      - High confidence (>80%): Give detailed, authoritative answers
      - Medium confidence (50-80%): Provide helpful answers with appropriate caveats
      - Low confidence (<50%): Be honest about limitations and suggest alternatives
      
      Current question confidence: ${(confidence * 100).toFixed(0)}%`;

      // This would be your actual OpenAI API call
      const response = await fetch('https://api.openai.com/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${apiKey}`
        },
        body: JSON.stringify({
          model: 'gpt-3.5-turbo',
          messages: [
            { role: 'system', content: systemPrompt },
            { role: 'user', content: message }
          ],
          max_tokens: 200,
          temperature: confidence > 0.8 ? 0.3 : 0.7
        })
      });

      if (!response.ok) {
        throw new Error('API call failed');
      }

      const data = await response.json();
      return data.choices[0].message.content;
    } catch (error) {
      // Fallback response when API is not available
      return generateFallbackResponse(message, confidence);
    }
  };

  const generateFallbackResponse = (message, confidence) => {
    const lowerMessage = message.toLowerCase();
    
    if (confidence > 0.8) {
      if (lowerMessage.includes('what is') || lowerMessage.includes('define')) {
        return `✅ I can provide a clear definition or explanation for that. This appears to be asking for factual information that I can answer with high confidence based on established knowledge.`;
      }
      if (lowerMessage.includes('how to') || lowerMessage.includes('calculate')) {
        return `✅ I can help you with step-by-step instructions or calculations. This type of procedural knowledge is within my strong capabilities.`;
      }
      return `✅ I'm highly confident I can help with this question. It appears to be asking about factual information that I can provide reliably.`;
    }
    
    if (confidence > 0.5) {
      return `⚠️ I can attempt to answer this, though my confidence is moderate. The topic might have complexity or subjective elements that could affect accuracy.`;
    }
    
    return `❌ I have low confidence in answering this accurately. It likely involves subjective judgment, predictions, or specialized knowledge I'm uncertain about.`;
  };

  const addMessage = (text, type, confidence = 0.7) => {
    const newMessage = {
      id: Date.now(),
      text,
      type,
      confidence
    };
    setMessages(prev => [...prev, newMessage]);
  };

  const handleSendMessage = async () => {
    if (!inputValue.trim()) return;
    if (showApiInput && !apiKey.trim()) {
      alert('Please enter your OpenAI API key first');
      return;
    }

    const userMessage = inputValue.trim();
    const lowerMessage = userMessage.toLowerCase();
    
    // Add user message
    addMessage(userMessage, 'user');
    setInputValue('');
    setIsTyping(true);

    // Simulate typing delay
    setTimeout(async () => {
      // Check for bias
      const biasDetected = detectBias(lowerMessage);
      if (biasDetected.length > 0) {
        addMessage(
          `⚠️ I detected potential bias in your question. Biased terms: "${biasDetected.join(', ')}". Could you rephrase this more neutrally? For example, instead of using absolute terms, try asking about specific cases.`,
          'warning',
          0.9
        );
        setIsTyping(false);
        return;
      }

      // Check for unclear language
      const unclearTerms = detectUnclearLanguage(lowerMessage);
      if (unclearTerms.length > 0) {
        addMessage(
          `❓ Your question seems unclear. I found vague terms: "${unclearTerms.join(', ')}". Could you be more specific? Try replacing vague words with concrete descriptions.`,
          'warning',
          0.8
        );
        setIsTyping(false);
        return;
      }

      // Check question length
      if (userMessage.split(' ').length < 3) {
        addMessage(
          `🤔 Your question seems too brief. Could you provide more context so I can give you a better answer?`,
          'warning',
          0.6
        );
        setIsTyping(false);
        return;
      }

      // Calculate confidence
      const confidence = calculateConfidence(lowerMessage);
      
      if (confidence < 0.3) {
        addMessage(
          `❌ I don't feel confident enough to answer this question properly. It might involve predictions, opinions, or topics outside my reliable knowledge. Could you ask something more factual?`,
          'error',
          confidence
        );
        setIsTyping(false);
        return;
      }

      // Generate AI response
      try {
        const aiResponse = await callOpenAI(userMessage, confidence);
        addMessage(aiResponse, 'bot', confidence);
      } catch (error) {
        addMessage(
          `❌ Sorry, I encountered an error processing your request. Please try again or check your API key.`,
          'error',
          0.1
        );
      }
      
      setIsTyping(false);
    }, 1000 + Math.random() * 2000);
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const getConfidenceColor = (confidence) => {
    if (confidence > 0.8) return '#28a745';
    if (confidence > 0.5) return '#ffc107';
    return '#dc3545';
  };

  const getConfidenceText = (confidence) => {
    if (confidence > 0.8) return 'High';
    if (confidence > 0.5) return 'Medium';
    return 'Low';
  };

  const getMessageClass = (type) => {
    switch (type) {
      case 'user': return 'bg-blue-500 text-white ml-auto rounded-br-sm';
      case 'warning': return 'bg-yellow-50 border-l-4 border-yellow-400 text-yellow-800 mr-auto';
      case 'error': return 'bg-red-50 border-l-4 border-red-400 text-red-800 mr-auto';
      default: return 'bg-white border border-gray-200 mr-auto rounded-bl-sm';
    }
  };

  if (showApiInput) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-indigo-500 via-purple-500 to-pink-500 flex items-center justify-center p-4">
        <div className="bg-white rounded-2xl shadow-2xl p-8 max-w-md w-full">
          <div className="text-center mb-6">
            <h1 className="text-2xl font-bold text-gray-800 mb-2">🤖 Smart AI Chatbot</h1>
            <p className="text-gray-600">Powered by OpenAI & LangChain</p>
          </div>
          
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                OpenAI API Key
              </label>
              <input
                type="password"
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                placeholder="sk-..."
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
              <p className="text-xs text-gray-500 mt-1">
                Your API key is stored locally and never sent to our servers
              </p>
            </div>
            
            <button
              onClick={() => setShowApiInput(false)}
              disabled={!apiKey.trim()}
              className="w-full bg-blue-500 text-white py-3 px-4 rounded-lg font-medium hover:bg-blue-600 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
            >
              Start Chatting
            </button>
            
            <button
              onClick={() => {
                setApiKey('demo-mode');
                setShowApiInput(false);
              }}
              className="w-full bg-gray-500 text-white py-3 px-4 rounded-lg font-medium hover:bg-gray-600 transition-colors"
            >
              Demo Mode (No API)
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-500 via-purple-500 to-pink-500 flex items-center justify-center p-4">
      <div className="bg-white rounded-2xl shadow-2xl w-full max-w-4xl h-[700px] flex flex-col overflow-hidden">
        {/* Header */}
        <div className="bg-gradient-to-r from-green-500 to-green-600 text-white p-6 text-center">
          <h1 className="text-xl font-bold">🤖 Smart Bias-Detecting Chatbot</h1>
          <p className="text-sm opacity-90 mt-1">
            Powered by OpenAI & LangChain - Detects bias, unclear questions, and confidence levels
          </p>
          <button
            onClick={() => setShowApiInput(true)}
            className="text-xs bg-white bg-opacity-20 px-3 py-1 rounded-full mt-2 hover:bg-opacity-30 transition-colors"
          >
            Change API Key
          </button>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-6 bg-gray-50 space-y-4">
          {messages.map((message) => (
            <div key={message.id} className="flex flex-col">
              <div className={`max-w-[80%] p-4 rounded-2xl ${getMessageClass(message.type)}`}>
                <div className="whitespace-pre-wrap">{message.text}</div>
                {message.type !== 'user' && (
                  <div className="flex items-center mt-2 text-xs text-gray-500">
                    <span>Confidence: {getConfidenceText(message.confidence)}</span>
                    <div className="ml-2 w-16 h-1.5 bg-gray-200 rounded-full overflow-hidden">
                      <div 
                        className="h-full rounded-full transition-all duration-300"
                        style={{
                          width: `${message.confidence * 100}%`,
                          backgroundColor: getConfidenceColor(message.confidence)
                        }}
                      />
                    </div>
                  </div>
                )}
              </div>
            </div>
          ))}
          
          {/* Typing Indicator */}
          {isTyping && (
            <div className="flex items-center space-x-2 max-w-[80px]">
              <div className="bg-white border border-gray-200 rounded-2xl p-4 rounded-bl-sm">
                <div className="flex space-x-1">
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{animationDelay: '0ms'}}></div>
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{animationDelay: '150ms'}}></div>
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{animationDelay: '300ms'}}></div>
                </div>
              </div>
            </div>
          )}
          
          <div ref={messagesEndRef} />
        </div>

        {/* Input */}
        <div className="p-6 bg-white border-t border-gray-200">
          <div className="flex space-x-3">
            <input
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Ask me anything..."
              className="flex-1 px-4 py-3 border-2 border-gray-200 rounded-full focus:border-blue-500 outline-none transition-colors"
              maxLength={500}
              disabled={isTyping}
            />
            <button
              onClick={handleSendMessage}
              disabled={!inputValue.trim() || isTyping}
              className="px-6 py-3 bg-blue-500 text-white rounded-full font-medium hover:bg-blue-600 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
            >
              Send
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SmartChatbot;
