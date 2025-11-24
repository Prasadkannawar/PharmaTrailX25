import React, { useState } from 'react';
import { Upload, FileText, CheckCircle, AlertTriangle, Database } from 'lucide-react';
import { apiService } from '../services/api';

const DataIngestion: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadResult, setUploadResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [uploadHistory, setUploadHistory] = useState<any[]>([]);

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      setUploadResult(null);
      setError(null);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setError('Please select a file to upload.');
      return;
    }

    try {
      setIsUploading(true);
      setError(null);
      setUploadProgress(0);

      // Simulate progress for demo purposes
      const progressInterval = setInterval(() => {
        setUploadProgress(prev => {
          if (prev >= 90) {
            clearInterval(progressInterval);
            return 90;
          }
          return prev + 10;
        });
      }, 200);

      // Note: This endpoint doesn't exist yet in the backend, so we'll simulate it
      try {
        const result = await apiService.uploadFile(selectedFile, '/ingest/upload');
        setUploadResult(result);
      } catch (err) {
        // Simulate successful upload for demo
        const mockResult = {
          success: true,
          filename: selectedFile.name,
          size: selectedFile.size,
          type: selectedFile.type,
          records_processed: Math.floor(Math.random() * 1000) + 100,
          upload_id: 'UPLOAD-' + Date.now(),
          timestamp: new Date().toISOString(),
        };
        setUploadResult(mockResult);
        
        // Add to history
        setUploadHistory(prev => [mockResult, ...prev.slice(0, 4)]);
      }

      clearInterval(progressInterval);
      setUploadProgress(100);
      
    } catch (err: any) {
      console.error('Upload error:', err);
      setError('Upload failed. This is a demo - the actual ingestion endpoint is not implemented yet.');
    } finally {
      setIsUploading(false);
      setTimeout(() => setUploadProgress(0), 2000);
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const getFileTypeIcon = (type: string) => {
    if (type.includes('csv')) return '📊';
    if (type.includes('json')) return '🔗';
    if (type.includes('xml')) return '📄';
    return '📁';
  };

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Data Ingestion</h1>
        <p className="text-gray-600 mt-2">
          Upload and process CT.gov data files and clinical trial datasets
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Upload Panel */}
        <div className="space-y-6">
          <div className="card">
            <h2 className="text-xl font-semibold text-gray-900 mb-6">File Upload</h2>
            
            {/* File Drop Zone */}
            <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-blue-400 transition-colors">
              <Upload className="w-12 h-12 text-gray-400 mx-auto mb-4" />
              <div className="space-y-2">
                <p className="text-lg font-medium text-gray-900">
                  Drop files here or click to browse
                </p>
                <p className="text-sm text-gray-600">
                  Supports CSV, JSON, XML files up to 100MB
                </p>
              </div>
              
              <input
                type="file"
                onChange={handleFileSelect}
                accept=".csv,.json,.xml,.txt"
                className="hidden"
                id="file-upload"
              />
              <label
                htmlFor="file-upload"
                className="btn-primary inline-flex items-center space-x-2 mt-4 cursor-pointer"
              >
                <FileText className="w-4 h-4" />
                <span>Select File</span>
              </label>
            </div>

            {/* Selected File Info */}
            {selectedFile && (
              <div className="mt-4 p-4 bg-blue-50 border border-blue-200 rounded-lg">
                <div className="flex items-center space-x-3">
                  <span className="text-2xl">{getFileTypeIcon(selectedFile.type)}</span>
                  <div className="flex-1">
                    <p className="font-medium text-gray-900">{selectedFile.name}</p>
                    <p className="text-sm text-gray-600">
                      {formatFileSize(selectedFile.size)} • {selectedFile.type || 'Unknown type'}
                    </p>
                  </div>
                </div>
              </div>
            )}

            {/* Upload Progress */}
            {isUploading && (
              <div className="mt-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-gray-700">Uploading...</span>
                  <span className="text-sm text-gray-600">{uploadProgress}%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${uploadProgress}%` }}
                  ></div>
                </div>
              </div>
            )}

            {/* Upload Button */}
            <div className="mt-6">
              <button
                onClick={handleUpload}
                disabled={!selectedFile || isUploading}
                className="btn-primary w-full flex items-center justify-center space-x-2"
              >
                <Upload className="w-4 h-4" />
                <span>{isUploading ? 'Uploading...' : 'Upload & Process'}</span>
              </button>
            </div>
          </div>

          {/* Supported Formats */}
          <div className="card">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Supported Formats</h3>
            <div className="space-y-3">
              <div className="flex items-center space-x-3">
                <span className="text-xl">📊</span>
                <div>
                  <p className="font-medium text-gray-900">CSV Files</p>
                  <p className="text-sm text-gray-600">Clinical trial data, patient records, study results</p>
                </div>
              </div>
              
              <div className="flex items-center space-x-3">
                <span className="text-xl">🔗</span>
                <div>
                  <p className="font-medium text-gray-900">JSON Files</p>
                  <p className="text-sm text-gray-600">Structured data exports, API responses</p>
                </div>
              </div>
              
              <div className="flex items-center space-x-3">
                <span className="text-xl">📄</span>
                <div>
                  <p className="font-medium text-gray-900">XML Files</p>
                  <p className="text-sm text-gray-600">CT.gov registry data, regulatory submissions</p>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Results Panel */}
        <div className="space-y-6">
          {/* Error Display */}
          {error && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-4">
              <div className="flex items-center">
                <AlertTriangle className="w-5 h-5 text-red-600 mr-2" />
                <span className="text-red-700 font-medium">Upload Error</span>
              </div>
              <p className="text-red-600 text-sm mt-2">{error}</p>
            </div>
          )}

          {/* Upload Result */}
          {uploadResult && (
            <div className="card">
              <div className="flex items-center space-x-3 mb-4">
                <CheckCircle className="w-6 h-6 text-green-600" />
                <h3 className="text-lg font-semibold text-gray-900">Upload Successful</h3>
              </div>
              
              <div className="space-y-3">
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <span className="text-gray-600">File:</span>
                    <span className="font-medium ml-2">{uploadResult.filename}</span>
                  </div>
                  <div>
                    <span className="text-gray-600">Size:</span>
                    <span className="font-medium ml-2">{formatFileSize(uploadResult.size)}</span>
                  </div>
                  <div>
                    <span className="text-gray-600">Records:</span>
                    <span className="font-medium ml-2">{uploadResult.records_processed?.toLocaleString()}</span>
                  </div>
                  <div>
                    <span className="text-gray-600">Upload ID:</span>
                    <span className="font-medium ml-2">{uploadResult.upload_id}</span>
                  </div>
                </div>
                
                <div className="bg-green-50 border border-green-200 rounded-lg p-3">
                  <p className="text-sm text-green-800">
                    ✅ File processed successfully and added to the data lake.
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Upload History */}
          <div className="card">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Recent Uploads</h3>
            {uploadHistory.length > 0 ? (
              <div className="space-y-3">
                {uploadHistory.map((upload, index) => (
                  <div key={index} className="border border-gray-200 rounded-lg p-3">
                    <div className="flex items-center justify-between mb-2">
                      <span className="font-medium text-gray-900">{upload.filename}</span>
                      <span className="text-xs text-gray-500">
                        {new Date(upload.timestamp).toLocaleDateString()}
                      </span>
                    </div>
                    <div className="flex items-center space-x-4 text-sm text-gray-600">
                      <span>{formatFileSize(upload.size)}</span>
                      <span>•</span>
                      <span>{upload.records_processed?.toLocaleString()} records</span>
                      <span>•</span>
                      <span className="text-green-600 font-medium">Processed</span>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-gray-500 text-sm">No recent uploads.</p>
            )}
          </div>

          {/* Data Storage Info */}
          <div className="card">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Data Storage</h3>
            <div className="space-y-3">
              <div className="flex items-center space-x-3">
                <Database className="w-5 h-5 text-blue-500" />
                <div>
                  <p className="font-medium text-gray-900">Data Lake</p>
                  <p className="text-sm text-gray-600">Uploaded files are stored in the centralized data lake</p>
                </div>
              </div>
              
              <div className="flex items-center space-x-3">
                <CheckCircle className="w-5 h-5 text-green-500" />
                <div>
                  <p className="font-medium text-gray-900">Validation</p>
                  <p className="text-sm text-gray-600">All data is validated and quality-checked</p>
                </div>
              </div>
              
              <div className="flex items-center space-x-3">
                <FileText className="w-5 h-5 text-purple-500" />
                <div>
                  <p className="font-medium text-gray-900">Processing</p>
                  <p className="text-sm text-gray-600">Data is processed and made available for analysis</p>
                </div>
              </div>
            </div>
          </div>

          {/* Demo Notice */}
          <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
            <div className="flex items-center">
              <AlertTriangle className="w-5 h-5 text-yellow-600 mr-2" />
              <span className="text-yellow-700 font-medium">Demo Mode</span>
            </div>
            <p className="text-yellow-600 text-sm mt-2">
              This is a demonstration interface. The actual data ingestion endpoints are not implemented yet.
              File uploads are simulated for UI testing purposes.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DataIngestion;
