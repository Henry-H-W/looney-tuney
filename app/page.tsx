import Image from "next/image";
import { Button } from "@/app/components/ui/button";
import { Input } from "@/app/components/ui/input";
import { Play } from "lucide-react";

export default function Home() {
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
          />
          <Button variant="secondary" className="whitespace-nowrap">
            Choose File
          </Button>
        </div>

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

        {/* Play Button */}
        <Button
          size="icon"
          className="h-16 w-16 rounded-full bg-white hover:bg-gray-200 text-black"
        >
          <Play className="h-8 w-8" />
        </Button>

        {/* Footer Line */}
        <div className="w-full max-w-2xl border-t border-dotted border-gray-800 pt-8 mt-8" />
      </div>
    </main>
  );
}
