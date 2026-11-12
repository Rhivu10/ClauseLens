import { useState } from "react";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import Nav from "@/components/Nav";
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