import { useState } from "react";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import Nav from "@/components/Nav";
import LineWaves from "@/components/ui/LineWaves";
import IntroSplash from "@/components/IntroSplash";
import LandingPage from "@/pages/LandingPage";
import UploadPage from "@/pages/UploadPage";
import ComparePage from "@/pages/ComparePage";
import ConsultPage from "@/pages/ConsultPage";

export default function App() {
  const [introDone, setIntroDone] = useState(false);

  return (
    <BrowserRouter>
      {!introDone && <IntroSplash onDone={() => setIntroDone(true)} />}
      {/* flowing lines behind every page, in the ClauseLens colours */}
      <LineWaves
        lightMode
        color1="#882E3A"
        color2="#687C86"
        color3="#3A1F1F"
        opacity={0.35}
        speed={0.2}
        lineCount={60}
        lineWidth={1.1}
        edgeFade={0.4}
        className="pointer-events-none"
        style={{ position: "fixed", inset: 0, zIndex: -1 }}
      />
      <Nav />
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/upload" element={<UploadPage />} />
        <Route path="/compare" element={<ComparePage />} />
        <Route path="/consult" element={<ConsultPage />} />
        <Route path="/chat" element={<Navigate to="/consult" replace />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
}