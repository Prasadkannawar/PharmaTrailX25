import React, { useState, useEffect } from 'react';
import { Shield, Clock, Hash, RefreshCw } from 'lucide-react';
import { apiService, BlockchainEvent } from '../services/api';

const BlockchainViewer: React.FC = () => {
  const [blockchainData, setBlockchainData] = useState<{
    blocks: BlockchainEvent[];
    stats: any;
  } | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [limit, setLimit] = useState(10);

  const fetchBlockchainData = async () => {
    try {
      setIsLoading(true);
      setError(null);
      
      const result = await apiService.getBlockchainChain(limit);
      setBlockchainData(result);
      
    } catch (err: any) {
      console.error('Blockchain fetch error:', err);
      setError(err.response?.data?.detail || 'Failed to fetch blockchain data.');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchBlockchainData();
  }, [limit]);

  const formatTimestamp = (timestamp: string) => {
    return new Date(timestamp).toLocaleString();
  };

  const getEventTypeColor = (eventType?: string) => {
    const type = (eventType || 'unknown').toLowerCase();
    switch (type) {
      case 'prediction':
        return 'bg-blue-100 text-blue-800';
      case 'nlp_processing':
        return 'bg-green-100 text-green-800';
      case 'digital_twin_simulation':
        return 'bg-purple-100 text-purple-800';
      case 'scenario_analysis':
        return 'bg-yellow-100 text-yellow-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Blockchain Audit Trail</h1>
        <p className="text-gray-600 mt-2">
          Immutable audit log of all system events and transactions
        </p>
      </div>

      {/* Controls */}
      <div className="card">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Show Latest
              </label>
              <select
                value={limit}
                onChange={(e) => setLimit(Number(e.target.value))}
                className="input-field w-24"
              >
                <option value={5}>5</option>
                <option value={10}>10</option>
                <option value={25}>25</option>
                <option value={50}>50</option>
              </select>
            </div>
            
            {blockchainData && (
              <div className="text-sm text-gray-600">
                <div>Total Blocks: <span className="font-semibold">{blockchainData.stats?.total_blocks || 0}</span></div>
                <div>Chain Integrity: 
                  <span className={`font-semibold ml-1 ${blockchainData.stats?.integrity_verified ? 'text-green-600' : 'text-red-600'}`}>
                    {blockchainData.stats?.integrity_verified ? 'Verified' : 'Failed'}
                  </span>
                </div>
              </div>
            )}
          </div>
          
          <button
            onClick={fetchBlockchainData}
            disabled={isLoading}
            className="btn-secondary flex items-center space-x-2"
          >
            <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
            <span>Refresh</span>
          </button>
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <div className="flex items-center">
            <Shield className="w-5 h-5 text-red-600 mr-2" />
            <span className="text-red-700 font-medium">Blockchain Error</span>
          </div>
          <p className="text-red-600 text-sm mt-2">{error}</p>
        </div>
      )}

      {/* Blockchain Events */}
      {blockchainData && blockchainData.blocks.length > 0 && (
        <div className="space-y-4">
          {blockchainData.blocks.map((block, index) => (
            <div key={block.index} className="card">
              <div className="flex items-start justify-between mb-4">
                <div className="flex items-center space-x-3">
                  <div className="w-8 h-8 bg-blue-100 rounded-lg flex items-center justify-center">
                    <Shield className="w-5 h-5 text-blue-600" />
                  </div>
                  <div>
                    <h3 className="font-semibold text-gray-900">Block #{block.index}</h3>
                    <p className="text-sm text-gray-600 flex items-center">
                      <Clock className="w-3 h-3 mr-1" />
                      {formatTimestamp(block.timestamp)}
                    </p>
                  </div>
                </div>
                
                <span className={`status-badge ${getEventTypeColor(block.data.event_type)}`}>
                  {block.data.event_type || 'Unknown Event'}
                </span>
              </div>

              <div className="space-y-3">
                {/* Event Data */}
                <div>
                  <h4 className="text-sm font-medium text-gray-900 mb-2">Event Data</h4>
                  <div className="bg-gray-50 rounded-lg p-3">
                    <div className="space-y-1 text-sm">
                      {block.data.trial_id && (
                        <div>
                          <span className="text-gray-600">Trial ID:</span>
                          <span className="font-medium ml-2">{block.data.trial_id}</span>
                        </div>
                      )}
                      
                      {Object.entries(block.data).map(([key, value]) => {
                        if (key === 'event_type' || key === 'trial_id') return null;
                        return (
                          <div key={key}>
                            <span className="text-gray-600">{key.replace(/_/g, ' ')}:</span>
                            <span className="font-medium ml-2">
                              {typeof value === 'object' ? JSON.stringify(value, null, 2) : String(value)}
                            </span>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                </div>

                {/* Hash Information */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <h4 className="text-sm font-medium text-gray-900 mb-1">Block Hash</h4>
                    <div className="flex items-center space-x-2">
                      <Hash className="w-3 h-3 text-gray-400" />
                      <code className="text-xs text-gray-600 bg-gray-100 px-2 py-1 rounded">
                        {block.hash.substring(0, 20)}...
                      </code>
                    </div>
                  </div>
                  
                  <div>
                    <h4 className="text-sm font-medium text-gray-900 mb-1">Previous Hash</h4>
                    <div className="flex items-center space-x-2">
                      <Hash className="w-3 h-3 text-gray-400" />
                      <code className="text-xs text-gray-600 bg-gray-100 px-2 py-1 rounded">
                        {block.previous_hash.substring(0, 20)}...
                      </code>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Empty State */}
      {blockchainData && blockchainData.blocks.length === 0 && (
        <div className="card text-center py-12">
          <Shield className="w-12 h-12 text-gray-400 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">No Blockchain Events</h3>
          <p className="text-gray-600">
            No events have been logged to the blockchain yet.
          </p>
        </div>
      )}

      {/* Loading State */}
      {isLoading && !blockchainData && (
        <div className="card text-center py-12">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading blockchain data...</p>
        </div>
      )}
    </div>
  );
};

export default BlockchainViewer;
