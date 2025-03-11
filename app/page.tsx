"use client";

import Image from "next/image";
import { Button } from "@/app/components/ui/button";
import { Input } from "@/app/components/ui/input";
import { Play } from "lucide-react";
import { useState } from "react";
import * as Tone from "tone";
import { Midi } from "@tonejs/midi";

export default function Home() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [uploadMessage, setUploadMessage] = useState<string>("");
  const [generatedFileUrl, setGeneratedFileUrl] = useState<string | null>(null);
  const [generatedFileName, setGeneratedFileName] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files.length > 0) {
      setSelectedFile(event.target.files[0]);
      setErrorMessage(null); // Reset errors on file change
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setUploadMessage("Please select a file first.");
      return;
    }

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await fetch("http://localhost:5000/upload", {
        method: "POST",
        body: formData,
      });

      if (response.ok) {
        setUploadMessage("File uploaded successfully! You can now generate music.");
        setErrorMessage(null);
      } else {
        const errorData = await response.json();
        setUploadMessage("File upload failed.");
        setErrorMessage(errorData.error || "An unknown error occurred during upload.");
      }
    } catch (error) {
      setUploadMessage("Error uploading file.");
      setErrorMessage(error instanceof Error ? error.message : "An unknown network error occurred.");
    }
  };

  const handlePlay = async () => {
    if (!generatedFileUrl) return;

    const midi = await fetch(generatedFileUrl).then((res) => res.arrayBuffer());
    const parsed = new Midi(midi);
    const synth = new Tone.PolySynth(Tone.Synth).toDestination();

    parsed.tracks.forEach((track) => {
      track.notes.forEach((note) => {
        synth.triggerAttackRelease(note.name, note.duration, note.time);
      });
    });

    Tone.Transport.start();
  };

  const handleGenerate = async () => {
    if (!selectedFile) {
      setErrorMessage("Please upload a file before generating.");
      return;
    }

    try {
      const response = await fetch("http://localhost:5000/generate", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ filename: selectedFile.name }),  // Send the uploaded filename
      });

      const result = await response.json();

      if (response.ok) {
        setGeneratedFileUrl(`http://localhost:5000/output/${result.generated_filename}`);
        setGeneratedFileName(result.generated_filename);
        setUploadMessage("Music generated successfully!");
        setErrorMessage(null);
      } else {
        setUploadMessage("Music generation failed.");
        setErrorMessage(result.error || "An unknown error occurred during generation.");
      }
    } catch (error) {
      setUploadMessage("Error generating music.");
      setErrorMessage(error instanceof Error ? error.message : "An unknown network error occurred.");
    }
  };

  return (
    <main
      className="min-h-screen flex flex-col items-center justify-center px-4 relative overflow-hidden"
      style={{
        backgroundImage: "url('/assets/background.jpg')",
        backgroundSize: "cover",
        backgroundPosition: "center",
        backgroundRepeat: "no-repeat",
      }}
    >
      {/* Logo */}
      <div className="absolute top-8 left-8">
        <Image
          src="/assets/logo.png"
          alt="Music Logo"
          width={80}
          height={80}
          className="rounded-full"
        />
      </div>

      {/* Main Content */}
      <div className="flex flex-col items-center gap-8 max-w-2xl mx-auto text-center">
        <div className="space-y-4">
          <h1 className="text-4xl font-bold tracking-tighter text-white sm:text-5xl xl:text-6xl/none">
            Welcome to our AI-Powered Music Composer!
          </h1>
          <p className="text-lg text-gray-300">
            Start generating personalized music with any MIDI file!
          </p>
        </div>

        {/* File Upload */}
        <div className="w-full max-w-md flex gap-4">
          <Input
            type="file"
            className="bg-gray-900/50 border-gray-800 text-gray-400 file:bg-gray-800 file:text-white file:border-0 hover:file:bg-gray-700"
            accept=".midi,.mid"
            onChange={handleFileChange}
          />
          <Button variant="secondary" className="whitespace-nowrap" onClick={handleUpload}>
            Upload File
          </Button>
        </div>

        {/* Upload Message */}
        {uploadMessage && <p className="text-green-400">{uploadMessage}</p>}

        {/* Error Message Display */}
        {errorMessage && <p className="text-red-400">{errorMessage}</p>}

        {/* Style Selection */}
        <div className="space-y-4 w-full">
          <p className="text-lg font-medium text-white">Select a Style</p>
          <div className="flex flex-wrap justify-center gap-4">
            {["Classical", "Pop", "Jazz", "Rock", "Electronic", "Folk"].map(
              (style) => (
                <Button
                  key={style}
                  variant="outline"
                  className="min-w-[100px] bg-gray-900/50 border-gray-800 text-white hover:bg-gray-800 hover:text-white"
                >
                  {style}
                </Button>
              )
            )}
          </div>
        </div>

        {/* Generate Button */}
        <Button
          className="min-w-[100px] bg-gray-900/50 border-gray-800 text-white hover:bg-gray-800 hover:text-white"
          variant="primary"
          onClick={handleGenerate}
        >
          Generate Music
        </Button>

        {/* Play Button */}
        <Button
          size="icon"
          className="h-16 w-16 rounded-full bg-white hover:bg-gray-200 text-black"
          onClick={handlePlay}
          disabled={!generatedFileUrl}
        >
          <Play className="h-8 w-8" />
        </Button>

        {/* Section to Display Generated MIDI File */}
        {generatedFileUrl && (
          <div className="mt-4 p-4 bg-gray-800 rounded-lg shadow-lg text-white text-center w-full max-w-md">
            <h2 className="text-xl font-bold">Generated Music</h2>
            <p className="text-gray-400">Your AI-generated MIDI file:</p>

            {/* Play the Generated MIDI */}
            <audio controls className="w-full mt-2">
              <source src={generatedFileUrl} type="audio/midi" />
              Your browser does not support the audio element.
            </audio>

            {/* Download Link */}
            <a
              href={generatedFileUrl}
              download={generatedFileName || "generated_music.mid"}
              className="mt-2 inline-block bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600"
            >
              Download MIDI
            </a>
          </div>
        )}

        {/* Footer Line */}
        <div className="w-full max-w-2xl border-t border-dotted border-gray-800 pt-8 mt-8" />
      </div>
    </main>
  );
}
