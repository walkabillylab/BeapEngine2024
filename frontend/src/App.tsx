import React from "react";
import { BrowserRouter as Router, Route, Routes } from "react-router-dom";
import Navbar from "./navbar/Navbar";
import Home from "./Home";
import Logout from "./Logout";
import ProcessedDataPage from "./ProcessDataPage/ProcessedDataPage";
import PredictedFiles from "./PredictedFiles";
// import FileUploadPage from "./FileUploadPage/FileUploadPage";
import "./App.css";

function App(): React.ReactElement<typeof Router> {
  return (
    <Router>
      <Navbar />
      <Routes>
        {/* Public Routes */}
        <Route path="/" element={<Home />} />
        <Route path="/Logout" element={<Logout />} />
        {/* <Route path="/FileUploadPage" element={<FileUploadPage />} /> */}

        {/* Protected Routes */}
        {/* This tag is to be added when the implementation of authentication is to be created */}
        {/* <Route element={<RequireAuth type={"user"} />}></Route> */}
        <Route path="/ProcessedDataPage" element={<ProcessedDataPage />} />
        <Route path="/PredictedFiles" element={<PredictedFiles />} />
      </Routes>
    </Router>
  );
}

export default App;
