/**
 * Reverse geocode lat/lng to a human-readable address via Nominatim (free, no key).
 */
export const reverseGeocode = async (lat, lng) => {
  try {
    const res = await fetch(
      `https://nominatim.openstreetmap.org/reverse?lat=${lat}&lon=${lng}&format=json&addressdetails=1`,
      { headers: { 'Accept-Language': 'en' } }
    );
    if (!res.ok) return '';
    const data = await res.json();
    return data.display_name || '';
  } catch {
    return '';
  }
};

/**
 * Forward-search an address string to lat/lng via Nominatim.
 * Returns the first result or null.
 */
export const searchAddress = async (query) => {
  try {
    const res = await fetch(
      `https://nominatim.openstreetmap.org/search?q=${encodeURIComponent(query)}&format=json&limit=1&addressdetails=1`,
      { headers: { 'Accept-Language': 'en' } }
    );
    if (!res.ok) return null;
    const results = await res.json();
    if (!results.length) return null;
    return {
      lat: parseFloat(results[0].lat),
      lng: parseFloat(results[0].lon),
      displayName: results[0].display_name || '',
    };
  } catch {
    return null;
  }
};
