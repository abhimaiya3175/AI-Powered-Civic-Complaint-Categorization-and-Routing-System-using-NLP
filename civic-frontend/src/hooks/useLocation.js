import { useState, useCallback } from 'react';
import { reverseGeocode, searchAddress } from '../utils/location';

export const useLocation = () => {
  const [locationState, setLocationState] = useState({
    latitude: null,
    longitude: null,
    timestamp: null,
    accuracy: null,
    error: null,
    address: '',
    isLoading: false,
  });

  const getLocation = useCallback(async () => {
    setLocationState(prev => ({ ...prev, isLoading: true, error: null }));
    
    if (!navigator.geolocation) {
      setLocationState(prev => ({ ...prev, isLoading: false, error: 'Geolocation is not supported by your browser.' }));
      return null;
    }

    return new Promise((resolve, reject) => {
      navigator.geolocation.getCurrentPosition(
        async (position) => {
          const lat = position.coords.latitude;
          const lng = position.coords.longitude;
          const acc = position.coords.accuracy;
          const ts = new Date().toISOString();

          // Get human-readable address
          const addr = await reverseGeocode(lat, lng);

          const newLocation = {
            latitude: lat,
            longitude: lng,
            accuracy: acc,
            timestamp: ts,
            address: addr,
            error: null,
            isLoading: false,
          };
          
          setLocationState(newLocation);
          resolve(newLocation);
        },
        (err) => {
          let errorMessage = 'Failed to get location.';
          if (err.code === 1) errorMessage = 'Location access denied. Please enable GPS permissions.';
          else if (err.code === 2) errorMessage = 'Location unavailable. Try moving to an open area.';
          else if (err.code === 3) errorMessage = 'Location request timed out.';
          
          setLocationState(prev => ({ ...prev, isLoading: false, error: errorMessage }));
          reject(new Error(errorMessage));
        },
        { enableHighAccuracy: true, timeout: 15000, maximumAge: 0 }
      );
    });
  }, []);

  const searchAndSetLocation = async (query) => {
    setLocationState(prev => ({ ...prev, isLoading: true, error: null }));
    const result = await searchAddress(query);
    if (result) {
      const newLocation = {
        latitude: result.lat,
        longitude: result.lng,
        address: result.displayName,
        timestamp: new Date().toISOString(),
        error: null,
        isLoading: false,
      };
      setLocationState(prev => ({ ...prev, ...newLocation }));
      return newLocation;
    } else {
      setLocationState(prev => ({ ...prev, isLoading: false, error: 'Address not found' }));
      return null;
    }
  };

  return {
    ...locationState,
    getLocation,
    searchAndSetLocation,
  };
};
