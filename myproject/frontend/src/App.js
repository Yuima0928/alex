import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import './App.css';

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");

  const handleSend = async () => {
    if (!input) return;
    const newMessage = { id: messages.length + 1, text: input, from: 'user' };
    setMessages([...messages, newMessage]);

    const response = await fetch('http://localhost:5000/api/message', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ message: input }),
    });
    const data = await response.json();
    const botMessage = { id: messages.length + 2, text: data.message, from: 'bot' };
    setMessages(messages => [...messages, botMessage]);
    setInput("");
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>yuima-alex</h1>
        <div className="chat-box">
          {messages.map(msg => (
            <div key={msg.id} className={msg.from === 'user' ? 'user-msg' : 'bot-msg'}>
              {msg.from === 'bot' ? (
                <ReactMarkdown>{msg.text}</ReactMarkdown>
              ) : (
                <p>{msg.text}</p>
              )}
            </div>
          ))}
        </div>
        <input
          type="text"
          value={input}
          onChange={e => setInput(e.target.value)}
          placeholder="Type your message..."
        />
        <button onClick={handleSend}>Send</button>
      </header>
    </div>
  );
}

export default App;
