"use client";
import { useState } from "react";
import RecommendationList from "../components/RecommendationList";
import Loader from "../components/Loader";
import { getRecommendations } from "../lib/api";

export default function Home() {
  const [itemId, setItemId] = useState("");
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    const res = await getRecommendations(itemId);
    setData(res);
    setLoading(false);
  };

  return (
    <main className="flex flex-col items-center p-10">
      <h1 className="text-2xl font-semibold mb-4">Magento Product Recommendor AI</h1>
      <form onSubmit={handleSubmit} className="flex gap-3 mb-6">
        <input
          type="number"
          placeholder="Enter Item ID"
          value={itemId}
          onChange={(e) => setItemId(e.target.value)}
          className="border px-3 py-2 rounded"
          required
        />
        <button className="bg-blue-600 text-white px-4 py-2 rounded">Fetch</button>
      </form>
      {loading ? <Loader /> : data && <RecommendationList data={data.recommendations} />}
    </main>
  );
}