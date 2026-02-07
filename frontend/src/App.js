import React, { useState, useEffect, useCallback, useRef } from "react";
import {
  Briefcase,
  Zap,
  MessageCircle,
  GitBranch,
  CheckCircle,
  Loader,
  Layout, 
  User, 
  Video 
} from "lucide-react";

const API_BASE_URL = "http://127.0.0.1:5000/api/v1";

// --- NEW COMPONENT: Highlight Transcript Issues ---
const TranscriptHighlighter = ({ analysis }) => {
  if (!analysis || !analysis.text) return null;

  const { text, issues } = analysis;
  let lastIndex = 0;
  const elements = [];

  issues.forEach((issue, index) => {
    // Text before the issue
    if (issue.start > lastIndex) {
      elements.push(
        <span key={`text-${index}`}>
          {text.substring(lastIndex, issue.start)}
        </span>
      );
    }

    // The issue itself (highlighted)
    elements.push(
      <span 
        key={`issue-${index}`} 
        className={`px-1 mx-0.5 rounded text-xs font-bold text-white cursor-help ${
          issue.type === 'filler' ? 'bg-yellow-500' : 'bg-red-500'
        }`}
        title={`${issue.feedback} (${issue.type})`}
      >
        {text.substring(issue.start, issue.end)}
      </span>
    );

    lastIndex = issue.end;
  });

  // Remaining text
  if (lastIndex < text.length) {
    elements.push(
      <span key="text-end">{text.substring(lastIndex)}</span>
    );
  }

  return (
    <div className="mt-4 p-4 bg-gray-50 border rounded-lg">
      <h4 className="font-semibold text-gray-700 mb-2 flex items-center">
        <MessageCircle size={16} className="mr-2" />
        Auto-Transcript Preview
      </h4>
      <p className="text-gray-800 leading-relaxed text-sm">
        {elements}
      </p>
      <div className="mt-2 flex gap-3 text-xs text-gray-500">
        <span className="flex items-center"><span className="w-2 h-2 bg-yellow-500 rounded-full mr-1"></span> Filler Word</span>
        <span className="flex items-center"><span className="w-2 h-2 bg-red-500 rounded-full mr-1"></span> Repetition</span>
      </div>
    </div>
  );
};

// --- MOCK DATA ---
const INTERVIEW_QUESTIONS = [
  "Tell me about a time you faced a technical challenge.",
  "What are your strengths and weaknesses?",
  "Explain a project you are proud of."
];

// --- UI HELPERS ---
const NavButton = ({ view, icon: Icon, label, currentView, setCurrentView }) => (
  <button
    onClick={() => setCurrentView(view)}
    className={`flex flex-col items-center justify-center p-3 rounded-lg transition-colors w-full mb-2 ${
      currentView === view
        ? "bg-blue-600 text-white shadow-lg"
        : "text-gray-600 hover:bg-gray-100"
    }`}
  >
    <Icon size={24} />
    <span className="text-xs mt-1 font-medium">{label}</span>
  </button>
);

const Card = ({ title, children, icon: Icon }) => (
  <div className="bg-white p-6 rounded-xl shadow-lg border border-gray-100 h-full">
    <h2 className="text-xl font-bold mb-4 text-gray-800 flex items-center">
      {Icon && <Icon className="mr-2 text-blue-500" size={24} />}
      {title}
    </h2>
    {children}
  </div>
);

const ProgressBar = ({ percentage, label, colorClass }) => (
  <div className="mb-4">
    <div className="flex justify-between mb-1 text-sm font-medium">
      <span>{label}</span>
      <span>{percentage}%</span>
    </div>
    <div className="w-full bg-gray-200 rounded-full h-2.5">
      <div
        className={`h-2.5 rounded-full transition-all duration-500 ${colorClass}`}
        style={{ width: `${percentage}%` }}
      />
    </div>
  </div>
);

// --- VIEW 1: PROFILE ---
const ProfileView = ({
  resumeText, setResumeText, analysisResult, profileMatchPercentage, recommendedDomain, isLoading, handleAnalyzeProfile
}) => (
  <div className="grid md:grid-cols-2 gap-8">
    <Card title="Student Profile Input" icon={Briefcase}>
      <textarea
        value={resumeText}
        onChange={(e) => setResumeText(e.target.value)}
        rows={10}
        placeholder="Paste your resume, skills or summary..."
        className="w-full p-3 border rounded-lg focus:ring-2 focus:ring-blue-500 outline-none"
        disabled={isLoading}
      />
      <button
        onClick={handleAnalyzeProfile}
        disabled={isLoading}
        className="mt-4 w-full bg-blue-500 hover:bg-blue-700 text-white font-semibold py-2 rounded-lg flex items-center justify-center transition-colors"
      >
        {isLoading ? <Loader className="animate-spin mr-2" size={18} /> : <Zap size={18} className="mr-2" />}
        {isLoading ? "Analyzing..." : "Analyze Profile & Skills"}
      </button>

      {analysisResult && (
        <div className="mt-4 p-3 bg-green-50 border border-green-200 text-green-700 rounded-lg text-sm flex items-center">
          <CheckCircle size={16} className="inline mr-2" /> Analysis complete.
        </div>
      )}
    </Card>
    <Card title="Career & Skill Gap Analysis" icon={GitBranch}>
      {analysisResult ? (
        <>
          <p className="text-lg font-bold text-green-600 mb-2">Recommended Domain: {recommendedDomain}</p>
          <ProgressBar
            percentage={profileMatchPercentage}
            label={`Profile Match Score (${recommendedDomain})`}
            colorClass={profileMatchPercentage > 75 ? "bg-green-500" : profileMatchPercentage > 50 ? "bg-yellow-500" : "bg-red-500"}
          />
          <h3 className="font-semibold mt-6 mb-2">Missing Skills:</h3>
          <div className="flex flex-wrap gap-2">
            {analysisResult.skill_gap_analysis?.missing_skills?.length ? (
              analysisResult.skill_gap_analysis.missing_skills.map((s, i) => (
                <span key={i} className="px-3 py-1 bg-red-100 text-red-700 rounded-full text-sm">{s}</span>
              ))
            ) : (
              <p className="text-sm text-gray-500">No major gaps 🎉</p>
            )}
          </div>
        </>
      ) : (
        <p className="text-gray-500">Analyze profile to see results.</p>
      )}
    </Card>
  </div>
);

// --- VIEW 2: INTERVIEW ---
const InterviewView = ({ interviewResult, setInterviewResult }) => {
  const [phase, setPhase] = useState("idle");
  const [isEvaluating, setIsEvaluating] = useState(false);
  const [seconds, setSeconds] = useState(0); 
  const [isSpeaking, setIsSpeaking] = useState(false);
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const recordedChunksRef = useRef([]);
  const [videoURL, setVideoURL] = useState(null);

  const startInterview = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
      streamRef.current = stream;
      recordedChunksRef.current = [];
      const recorder = new MediaRecorder(stream);
      mediaRecorderRef.current = recorder;
      recorder.ondataavailable = (e) => e.data.size && recordedChunksRef.current.push(e.data);
      recorder.start();
      setPhase("live");
    } catch (err) {
      console.error("Error accessing media devices:", err);
      alert("Could not access camera/microphone");
    }
  };

  useEffect(() => {
    if (phase === "live" && videoRef.current && streamRef.current) {
      videoRef.current.srcObject = streamRef.current;
    }
  }, [phase]);

  useEffect(() => {
    if (phase !== "live") return;
    const interval = setInterval(() => { setSeconds(prev => prev + 1); }, 1000);
    return () => clearInterval(interval);
  }, [phase]);

  useEffect(() => {
    if (phase !== "live" || !streamRef.current) return;
    const audioContext = new AudioContext();
    const source = audioContext.createMediaStreamSource(streamRef.current);
    const analyser = audioContext.createAnalyser();
    analyser.fftSize = 512;
    source.connect(analyser);
    const data = new Uint8Array(analyser.frequencyBinCount);
    let rafId;
    const detectSpeech = () => {
      analyser.getByteFrequencyData(data);
      const volume = data.reduce((a, b) => a + b, 0) / data.length;
      setIsSpeaking(volume > 20); 
      rafId = requestAnimationFrame(detectSpeech);
    };
    detectSpeech();
    return () => { cancelAnimationFrame(rafId); audioContext.close(); };
  }, [phase]);

  const endInterview = () => {
    if (!mediaRecorderRef.current) return;
    mediaRecorderRef.current.onstop = () => {
        const blob = new Blob(recordedChunksRef.current, { type: "video/webm" });
        const url = URL.createObjectURL(blob);
        setVideoURL(url);
        setPhase("ended");
        setSeconds(0);
    };
    mediaRecorderRef.current.stop();
    streamRef.current?.getTracks().forEach(track => track.stop());
  };

  const downloadReport = async () => {
  if (!interviewResult) return;

  const res = await fetch(`${API_BASE_URL}/download_interview_report`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(interviewResult),
  });

  const blob = await res.blob();
  const url = window.URL.createObjectURL(blob);

  const a = document.createElement("a");
  a.href = url;
  a.download = "AI_Interview_Report.pdf";
  a.click();

  window.URL.revokeObjectURL(url);
};

  const evaluatePerformance = async () => {
    if (!recordedChunksRef.current.length) { alert("No recording found"); return; }
    try {
        setIsEvaluating(true);
        const blob = new Blob(recordedChunksRef.current, { type: "video/webm" });
        const formData = new FormData();
        formData.append("video", blob, "interview.webm");
        
        const res = await fetch(`${API_BASE_URL}/mock_facial_interview`, {
            method: "POST",
            body: formData,
        });
        if (!res.ok) throw new Error("Backend failed");
        const data = await res.json();
        setInterviewResult(data);
        setPhase("result");
    } catch (err) {
        console.error(err);
        alert("Evaluation failed. Check backend.");
    } finally {
        setIsEvaluating(false);
    }
  };

  return (
    <div className="grid md:grid-cols-2 gap-8">
      <Card title="AI Mock Interview" icon={Zap}>
        {phase === "idle" && (
          <button onClick={startInterview} className="w-full bg-blue-600 text-white py-3 rounded-lg hover:bg-blue-700 transition-colors">
            🎥 Start Mock Interview
          </button>
        )}
        {phase === "live" && (
          <>
            <div className="relative">
                <video ref={videoRef} autoPlay playsInline muted className="w-full h-64 rounded-lg border mb-4 bg-black object-cover" />
                {isSpeaking && <div className="absolute top-2 right-2 bg-green-500 text-white text-xs px-2 py-1 rounded-full animate-pulse">Speaking</div>}
                <div className="absolute top-2 left-2 bg-red-500 text-white text-xs px-2 py-1 rounded-full">{new Date(seconds * 1000).toISOString().substr(14, 5)}</div>
            </div>
            <div className="bg-gray-50 border rounded-lg p-4 mb-4">
              <p className="font-semibold mb-2">Interview Questions:</p>
              <ul className="list-disc list-inside text-sm space-y-1">
                {INTERVIEW_QUESTIONS.map((q, i) => <li key={i}>{q}</li>)}
              </ul>
            </div>
            <button onClick={endInterview} className="w-full bg-red-500 text-white py-2 rounded-lg hover:bg-red-600 transition-colors">⏹ End Interview</button>
          </>
        )}
        {phase === "ended" && (
          <>
            <video src={videoURL} controls className="w-full rounded-lg border mb-4" />
            <button onClick={evaluatePerformance} disabled={isEvaluating} className="w-full bg-green-600 text-white py-2 rounded-lg hover:bg-green-700 transition-colors">
              {isEvaluating ? "Analyzing..." : "✅ Evaluate Performance"}
            </button>
          </>
        )}
      </Card>
      

      <Card title="Interview Performance Feedback" icon={Zap}>
        {phase === "result" && interviewResult ? (
          <>
          <button
  onClick={async () => {
    const res = await fetch(`${API_BASE_URL}/interview_report`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(interviewResult),
    });

    const blob = await res.blob();
    const url = window.URL.createObjectURL(blob);

    const a = document.createElement("a");
    a.href = url;
    a.download = "AI_Interview_Report.pdf";
    a.click();
    window.URL.revokeObjectURL(url);
  }}
  className="w-full mt-3 bg-indigo-600 text-white py-2 rounded-lg hover:bg-indigo-700"
>
  📄 Download Interview Report
</button>


            <h3 className="text-2xl font-bold text-blue-600 mb-2">Employability Score: {interviewResult.employability_score}</h3>
            <p><strong>Sentiment:</strong> {interviewResult.communication_analysis?.sentiment}</p>
            <p><strong>Dominant Emotion:</strong> {interviewResult.facial_analysis?.emotions?.dominant_emotion}</p>
            <p className="mt-2 text-gray-700">{interviewResult.facial_analysis?.feedback}</p>
            
            {/* --- THIS IS THE NEW PART THAT RENDERS THE TRANSCRIPT --- */}
            <TranscriptHighlighter analysis={interviewResult.communication_analysis?.transcript_analysis} />
          </>
          
        ) : (
          <div className="flex flex-col items-center justify-center h-full text-gray-500">
             <Zap className="mb-2 opacity-20" size={48} />
             <p>Complete interview and evaluate to see feedback.</p>
          </div>
        )}
      </Card>
    </div>
  );
};

// --- VIEW 3: MENTOR ---
const MentorView = ({ chatHistory, chatQuery, setChatQuery, handleChatQuery, isLoading, recommendedDomain }) => {
   return (
    <div className="grid md:grid-cols-2 gap-8">
        <Card title="Virtual Career Mentor" icon={MessageCircle}>
            <div className="flex items-center mb-3 text-sm text-gray-600">
                <MessageCircle size={16} className="mr-2 text-blue-500" />
                <span className="font-semibold text-blue-600">Virtual Career Mentor</span>
            </div>
            <div className="h-96 overflow-y-auto p-4 border border-gray-200 rounded-lg bg-gray-50 flex flex-col space-y-3">
                {chatHistory.map((msg, index) => (
                    <div key={index} className={`flex ${msg.type === 'user' ? 'justify-end' : 'justify-start'}`}>
                        <div className={`max-w-xs md:max-w-md p-3 rounded-xl shadow-md text-sm ${msg.type === 'user' ? 'bg-blue-500 text-white rounded-br-none' : 'bg-white text-gray-800 rounded-tl-none border border-gray-300 shadow-sm'}`}>
                            <div className="space-y-1" dangerouslySetInnerHTML={{ __html: msg.text.replace(/\n/g, '<br/>').replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>').replace(/Phase \d:/g, '<span class="font-semibold text-blue-600">$&</span>') }} />
                        </div>
                    </div>
                ))}
                {isLoading && <div className="flex justify-start"><div className="p-3 bg-white text-gray-500 rounded-xl rounded-tl-none border border-gray-300 text-sm"><Loader className="animate-spin inline mr-2" size={14} /> Mentor is typing...</div></div>}
            </div>
            <div className="mt-4 flex">
                <input type="text" value={chatQuery} onChange={(e) => setChatQuery(e.target.value)} placeholder={recommendedDomain === 'Unknown' ? "Analyze your profile first to chat..." : `Ask your mentor about ${recommendedDomain}, roadmap, interviews…`} className="flex-grow p-3 border border-gray-300 rounded-l-lg focus:ring-blue-500 focus:border-blue-500 text-sm outline-none" disabled={isLoading || recommendedDomain === 'Unknown'} onKeyPress={(e) => e.key === 'Enter' && handleChatQuery()} />
                <button onClick={handleChatQuery} className={`px-4 rounded-r-lg text-white font-semibold flex items-center justify-center transition-colors ${isLoading || recommendedDomain === 'Unknown' ? 'bg-gray-400' : 'bg-blue-500 hover:bg-blue-700'}`} disabled={isLoading || recommendedDomain === 'Unknown'}>Send</button>
            </div>
        </Card>
        <Card title="University Batch Analytics (Mock)" icon={Zap}>
           <div className="h-full space-y-4">
               <p className="text-gray-600 mb-4">This administrative view would provide batch-level insights into student readiness, visible only to faculty/admin staff.</p>
               <h4 className="font-bold text-gray-800">Key Metrics:</h4>
               <ul className="list-disc list-inside space-y-1 text-sm text-gray-700">
                   <li>Average Employability Score: <strong>78.5</strong> (Goal: 85+)</li>
                   <li>Top Skill Gap: <strong>Cloud Computing (35% of batch)</strong></li>
                   <li>Most Recommended Domain: <strong>Software Development</strong></li>
                   <li>Students Requiring Mentorship: <strong>12%</strong> (Score below 60)</li>
               </ul>
               <div className="mt-4 p-3 bg-indigo-50 border border-indigo-200 rounded-lg">
                   <p className="text-sm font-semibold text-indigo-700">Actionable Insight:</p>
                   <p className="text-xs text-indigo-600">Recommend a mandatory workshop on <strong>Cloud Computing</strong> to bridge the largest batch-level skill gap.</p>
               </div>
           </div>
       </Card>
    </div>
   );
};

// --- MAIN APP COMPONENT ---
export default function App() {
  const [interviewResult, setInterviewResult] = useState(null);
  const [resumeText, setResumeText] = useState('');
  const [analysisResult, setAnalysisResult] = useState(null);
  const [chatQuery, setChatQuery] = useState('');
  const [chatHistory, setChatHistory] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [currentView, setCurrentView] = useState('profile');
  
  const profileMatchPercentage = analysisResult?.profile_match_percentage || 0;
  const recommendedDomain = analysisResult?.career_recommendations?.[0] || 'Unknown';

  const handleAnalyzeProfile = useCallback(async () => {
    if (!resumeText.trim()) { setError("Please paste a resume or skill summary to analyze."); return; }
    setIsLoading(true); setError(null); setAnalysisResult(null);
    try {
      const response = await fetch(`${API_BASE_URL}/analyze_profile`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ resume_text: resumeText }), });
      if (!response.ok) throw new Error("Backend analysis failed.");
      const data = await response.json();
      setAnalysisResult(data); setInterviewResult(null);
      setChatHistory([{ type: 'mentor', text: `Hello! Based on your profile, your recommended domain is **${data.career_recommendations[0]}**. What can I help you with?`}]);
    } catch (err) { console.error("Analysis Error:", err); setError(`Failed to connect to backend or process data. Error: ${err.message}`); } finally { setIsLoading(false); }
  }, [resumeText]);

  const handleChatQuery = useCallback(async () => {
    if (!chatQuery.trim()) return;
    if (!recommendedDomain || recommendedDomain === 'Unknown') { setError("Please analyze your profile first to get a career domain."); return; }
    const newChatHistory = [...chatHistory, { type: 'user', text: chatQuery }];
    setChatHistory(newChatHistory); setChatQuery(''); setIsLoading(true); setError(null);
    try {
      const response = await fetch(`${API_BASE_URL}/mentor_chat`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ query: chatQuery, domain: recommendedDomain }), });
      if (!response.ok) throw new Error("Chatbot failed to respond.");
      const data = await response.json();
      setChatHistory(prev => [...prev, { type: 'mentor', text: data.mentor_response }]);
    } catch (err) { console.error("Chat Error:", err); setChatHistory(prev => [...prev, { type: 'mentor', text: "Sorry, I can't connect to my knowledge base right now. Please try again later." }]); } finally { setIsLoading(false); }
  }, [chatQuery, chatHistory, recommendedDomain]);    

  return (
    <div className="min-h-screen bg-gray-50 flex">
      <div className="w-24 bg-white border-r border-gray-200 flex flex-col items-center py-6 fixed h-full left-0 top-0 z-10">
        <div className="mb-8 bg-blue-100 p-2 rounded-xl"><Layout className="text-blue-600" size={28} /></div>
        <NavButton view="profile" icon={User} label="Profile" currentView={currentView} setCurrentView={setCurrentView} />
        <NavButton view="interview" icon={Video} label="Interview" currentView={currentView} setCurrentView={setCurrentView} />
        <NavButton view="mentor" icon={MessageCircle} label="Mentor" currentView={currentView} setCurrentView={setCurrentView} />
      </div>
      <div className="ml-24 flex-1 p-8">
        <div className="max-w-6xl mx-auto">
          <header className="mb-8">
            <h1 className="text-3xl font-bold text-gray-800">
                {currentView === 'profile' && 'Career Profile Analysis'}
                {currentView === 'interview' && 'AI Mock Interview'}
                {currentView === 'mentor' && 'Career Mentorship'}
            </h1>
            <p className="text-gray-500 mt-2">AI-driven insights for university students</p>
          </header>
          {error && <div className="mb-6 bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg flex items-center"><CheckCircle className="mr-2" size={20} />{error}</div>}
          {currentView === 'profile' && <ProfileView resumeText={resumeText} setResumeText={setResumeText} analysisResult={analysisResult} profileMatchPercentage={profileMatchPercentage} recommendedDomain={recommendedDomain} isLoading={isLoading} handleAnalyzeProfile={handleAnalyzeProfile} />}
          {currentView === 'interview' && <InterviewView interviewResult={interviewResult} setInterviewResult={setInterviewResult} />}
          {currentView === 'mentor' && <MentorView chatHistory={chatHistory} chatQuery={chatQuery} setChatQuery={setChatQuery} handleChatQuery={handleChatQuery} isLoading={isLoading} recommendedDomain={recommendedDomain} />}
        </div>
      </div>
    </div>
  );
}