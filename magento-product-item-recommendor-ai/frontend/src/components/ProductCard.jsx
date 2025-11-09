export default function ProductCard({ product }) {
  return (
    <div className="p-4 border rounded shadow-sm">
      <h2 className="font-semibold">{product.name}</h2>
      <p className="text-gray-600">₹{product.price}</p>
    </div>
  );
}