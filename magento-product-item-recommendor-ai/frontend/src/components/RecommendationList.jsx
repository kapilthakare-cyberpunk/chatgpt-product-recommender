import ProductCard from './ProductCard';

export default function RecommendationList({ data }) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
      {data.map((item) => (
        <ProductCard key={item.id} product={item} />
      ))}
    </div>
  );
}