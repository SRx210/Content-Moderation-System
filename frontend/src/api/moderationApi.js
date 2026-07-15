// src/api/moderationApi.js

import axios from "axios";

const API_BASE = import.meta.env.VITE_API_URL;

export async function moderateText(text) {
  const response = await axios.post(`${API_BASE}/moderate`, { text });
  return response.data;
}
