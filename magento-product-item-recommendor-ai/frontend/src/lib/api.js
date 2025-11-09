export const getRecommendations = async (itemId) => {
  try {
    const res = await fetch(`http://localhost:8000/recommend?item_id=${itemId}`);
    if (!res.ok) {
      throw new Error(`API request failed with status ${res.status}`);
    }
    return await res.json();
  } catch (error) {
    console.error('Error fetching recommendations:', error);
    throw error;
  }
};