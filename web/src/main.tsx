import React from "react";
import ReactDOM from "react-dom/client";
import { MantineProvider, createTheme } from "@mantine/core";
import "@mantine/core/styles.css";
import App from "./App";
import "./styles.css";

const theme = createTheme({
  primaryColor: "nornickel",
  primaryShade: 6,
  fontFamily: "Inter, Segoe UI, system-ui, sans-serif",
  defaultRadius: "md",
  headings: { fontFamily: "Inter, Segoe UI, system-ui, sans-serif", fontWeight: "700" },
  colors: {
    // Nornickel corporate palette per "Стандарт фирменный стиль" v1.1:
    // shade 6 = Pantone 3005 #0077C8 (основной синий), shade 8 = Pantone 2945 #004C97 (темно-синий)
    nornickel: [
      "#EAF4FC",
      "#D2E7F8",
      "#A6CFF1",
      "#79B7EA",
      "#4D9FE3",
      "#2087DC",
      "#0077C8",
      "#00629F",
      "#004C97",
      "#00325F",
    ],
  },
});

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <MantineProvider theme={theme} defaultColorScheme="light">
      <App />
    </MantineProvider>
  </React.StrictMode>
);
