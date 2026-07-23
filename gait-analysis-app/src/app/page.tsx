"use client";

import React, { useState, useRef, useEffect } from 'react';
import { Upload, Play, Activity } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { PoseLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';

export default function GaitAnalyzer() {
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [chartData, setChartData] = useState<any[]>([]);

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const poseLandmarkerRef = useRef<PoseLandmarker | null>(null);

  useEffect(() => {
    async function initMediaPipe() {
      const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
      );
      const poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
          delegate: "GPU"
        },
        runningMode: "VIDEO",
        numPoses: 1
      });
      poseLandmarkerRef.current = poseLandmarker;
    }
    initMediaPipe();
  }, []);

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setVideoFile(file);
      setVideoUrl(URL.createObjectURL(file));
      setChartData([]);
      setProgress(0);
    }
  };

  const calculateAngle = (a: any, b: any, c: any) => {
    const radians = Math.atan2(c.y - b.y, c.x - b.x) - Math.atan2(a.y - b.y, a.x - b.x);
    let angle = Math.abs(radians * 180.0 / Math.PI);
    if (angle > 180.0) angle = 360 - angle;
    return angle;
  };

  const processVideo = async () => {
    if (!videoRef.current || !poseLandmarkerRef.current || !canvasRef.current) return;
    
    setIsProcessing(true);
    setChartData([]);
    setProgress(0);

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    video.currentTime = 0;
    await video.play();

    const newData: any[] = [];
    let lastVideoTime = -1;

    const processFrame = () => {
      if (video.paused || video.ended) {
        setIsProcessing(false);
        setChartData(newData);
        return;
      }

      if (video.currentTime !== lastVideoTime) {
        const poseLandmarkerResult = poseLandmarkerRef.current!.detectForVideo(video, performance.now());
        
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        if (poseLandmarkerResult.landmarks && poseLandmarkerResult.landmarks[0]) {
          const landmarks = poseLandmarkerResult.landmarks[0];
          
          const leftHip = landmarks[23];
          const leftKnee = landmarks[25];
          const leftAnkle = landmarks[27];
          
          const rightHip = landmarks[24];
          const rightKnee = landmarks[26];
          const rightAnkle = landmarks[28];

          let leftAngle = null;
          let rightAngle = null;

          if (leftHip.visibility && leftHip.visibility > 0.5 && leftKnee.visibility && leftKnee.visibility > 0.5) {
            leftAngle = calculateAngle(leftHip, leftKnee, leftAnkle);
          }
          if (rightHip.visibility && rightHip.visibility > 0.5 && rightKnee.visibility && rightKnee.visibility > 0.5) {
            rightAngle = calculateAngle(rightHip, rightKnee, rightAnkle);
          }

          newData.push({
            time: video.currentTime.toFixed(2),
            leftKnee: leftAngle ? Math.round(leftAngle) : null,
            rightKnee: rightAngle ? Math.round(rightAngle) : null,
          });

          // Draw skeleton
          ctx.strokeStyle = '#00FF00';
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.moveTo(leftHip.x * canvas.width, leftHip.y * canvas.height);
          ctx.lineTo(leftKnee.x * canvas.width, leftKnee.y * canvas.height);
          ctx.lineTo(leftAnkle.x * canvas.width, leftAnkle.y * canvas.height);
          ctx.stroke();

          ctx.strokeStyle = '#FF0000';
          ctx.beginPath();
          ctx.moveTo(rightHip.x * canvas.width, rightHip.y * canvas.height);
          ctx.lineTo(rightKnee.x * canvas.width, rightKnee.y * canvas.height);
          ctx.lineTo(rightAnkle.x * canvas.width, rightAnkle.y * canvas.height);
          ctx.stroke();
        }

        lastVideoTime = video.currentTime;
        setProgress((video.currentTime / video.duration) * 100);
      }
      
      requestAnimationFrame(processFrame);
    };

    requestAnimationFrame(processFrame);
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 p-8 font-sans">
      <div className="max-w-6xl mx-auto space-y-8">
        <header className="border-b border-slate-800 pb-6 flex items-center gap-3">
          <Activity className="w-8 h-8 text-blue-500" />
          <div>
            <h1 className="text-3xl font-bold tracking-tight">Gait Analyzer</h1>
            <p className="text-slate-400">Browser-based Video Gait Analysis</p>
          </div>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          <div className="lg:col-span-1 space-y-6">
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6">
              <h2 className="text-xl font-semibold mb-4">1. Upload Video</h2>
              <label className="flex flex-col items-center justify-center w-full h-32 border-2 border-slate-700 border-dashed rounded-xl cursor-pointer hover:bg-slate-800/50 transition">
                <div className="flex flex-col items-center justify-center pt-5 pb-6">
                  <Upload className="w-8 h-8 text-slate-400 mb-2" />
                  <p className="text-sm text-slate-400 font-medium">Click to upload video</p>
                  <p className="text-xs text-slate-500 mt-1">MP4, WEBM, MOV</p>
                </div>
                <input type="file" className="hidden" accept="video/*" onChange={handleFileUpload} />
              </label>
            </div>

            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6">
              <h2 className="text-xl font-semibold mb-4">2. Analyze</h2>
              <button 
                onClick={processVideo}
                disabled={!videoUrl || isProcessing}
                className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-slate-800 disabled:text-slate-500 text-white font-medium py-3 px-4 rounded-xl transition flex justify-center items-center gap-2"
              >
                {isProcessing ? (
                  <>Processing... {Math.round(progress)}%</>
                ) : (
                  <><Play className="w-5 h-5" /> Analyze Gait</>
                )}
              </button>
            </div>
          </div>

          <div className="lg:col-span-2 space-y-6">
            <div className="bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden relative aspect-video flex items-center justify-center bg-black">
              {videoUrl ? (
                <>
                  <video 
                    ref={videoRef} 
                    src={videoUrl} 
                    className="absolute inset-0 w-full h-full object-contain opacity-50"
                    crossOrigin="anonymous"
                    playsInline
                    muted
                    onLoadedMetadata={() => {
                      if (videoRef.current && canvasRef.current) {
                        canvasRef.current.width = videoRef.current.videoWidth;
                        canvasRef.current.height = videoRef.current.videoHeight;
                      }
                    }}
                  />
                  <canvas 
                    ref={canvasRef} 
                    className="absolute inset-0 w-full h-full object-contain pointer-events-none"
                  />
                </>
              ) : (
                <p className="text-slate-500">No video uploaded</p>
              )}
            </div>
          </div>
        </div>

        {chartData.length > 0 && (
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 mt-8">
            <h2 className="text-xl font-semibold mb-6">Knee Angles Over Time</h2>
            <div className="h-96">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="time" stroke="#94a3b8" />
                  <YAxis stroke="#94a3b8" domain={[0, 180]} />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b' }}
                    labelStyle={{ color: '#94a3b8' }}
                  />
                  <Legend />
                  <Line type="monotone" dataKey="leftKnee" stroke="#3b82f6" name="Left Knee" dot={false} strokeWidth={2} />
                  <Line type="monotone" dataKey="rightKnee" stroke="#ef4444" name="Right Knee" dot={false} strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
