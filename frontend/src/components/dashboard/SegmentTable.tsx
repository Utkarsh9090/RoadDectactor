import React, { useState } from 'react';
import { FileSpreadsheet, FileText, Search } from 'lucide-react';
import type { SegmentDetail } from '../../types';

interface SegmentTableProps {
  segments: SegmentDetail[];
  filename: string;
}

export const SegmentTable: React.FC<SegmentTableProps> = ({ segments, filename }) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [filterAction, setFilterAction] = useState<string>('all');

  const filteredSegments = segments.filter((s) => {
    const matchesSearch =
      s.segment_id.toString().includes(searchTerm) ||
      s.condition_category.toLowerCase().includes(searchTerm.toLowerCase()) ||
      s.rl_action_name.toLowerCase().includes(searchTerm.toLowerCase());

    const matchesAction =
      filterAction === 'all' ||
      (filterAction === 'resurface' && s.rl_recommended_action === 2) ||
      (filterAction === 'repair' && s.rl_recommended_action === 1) ||
      (filterAction === 'none' && s.rl_recommended_action === 0);

    return matchesSearch && matchesAction;
  });

  const exportJSON = () => {
    const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(segments, null, 2));
    const downloadAnchor = document.createElement('a');
    downloadAnchor.setAttribute('href', dataStr);
    downloadAnchor.setAttribute('download', `${filename.replace(/\.[^/.]+$/, '')}_inspection_report.json`);
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
  };

  const exportCSV = () => {
    const headers = ['Segment ID', 'Frame Start', 'Frame End', 'PCI Score', 'Condition', 'Digs', 'Patches', 'Cracks', 'Potholes', 'RL Action'];
    const rows = segments.map((s) => [
      s.segment_id,
      s.frame_start,
      s.frame_end,
      s.pci_score,
      s.condition_category,
      s.defects.digs,
      s.defects.patches,
      s.defects.cracks,
      s.defects.potholes,
      `"${s.rl_action_name}"`,
    ]);

    const csvContent = 'data:text/csv;charset=utf-8,' + [headers.join(','), ...rows.map((e) => e.join(','))].join('\n');
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', `${filename.replace(/\.[^/.]+$/, '')}_segment_log.csv`);
    document.body.appendChild(link);
    link.click();
    link.remove();
  };

  return (
    <div className="glass-surface-static p-6 space-y-4">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <span className="eyebrow-label">Granular Breakdown</span>
          <h3 className="font-heading font-bold text-lg text-[#2A1418]">
            100-Meter Road Segment Log ({segments.length} Segments Analyzed)
          </h3>
        </div>

        {/* Export Buttons */}
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={exportJSON}
            className="btn-outline text-xs py-2 px-3 flex items-center gap-1.5"
            title="Download full JSON dataset"
          >
            <FileText className="w-4 h-4 text-[#9B4A57]" />
            <span>Export JSON</span>
          </button>
          <button
            type="button"
            onClick={exportCSV}
            className="btn-outline text-xs py-2 px-3 flex items-center gap-1.5"
            title="Download CSV spreadsheet"
          >
            <FileSpreadsheet className="w-4 h-4 text-[#9B4A57]" />
            <span>Export CSV</span>
          </button>
        </div>
      </div>

      {/* Filter and Search Bar */}
      <div className="flex flex-col sm:flex-row items-center gap-3 pt-2">
        <div className="relative w-full sm:w-72">
          <Search className="w-4 h-4 text-[#7C6A6D] absolute left-3 top-1/2 -translate-y-1/2" />
          <input
            type="text"
            placeholder="Search segment ID or status..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="morphism-input pl-9 pr-3 py-1.5 text-xs w-full"
          />
        </div>

        <div className="flex items-center gap-1.5 w-full sm:w-auto text-xs overflow-x-auto pb-1 sm:pb-0">
          <button
            type="button"
            onClick={() => setFilterAction('all')}
            className={`px-3 py-1.5 rounded-lg font-medium transition-all ${
              filterAction === 'all'
                ? 'bg-[#9B4A57] text-white shadow-sm'
                : 'bg-white text-[#7C6A6D] border border-[#9B4A57]/15 hover:bg-white/80'
            }`}
          >
            All Actions
          </button>
          <button
            type="button"
            onClick={() => setFilterAction('resurface')}
            className={`px-3 py-1.5 rounded-lg font-medium transition-all ${
              filterAction === 'resurface'
                ? 'bg-red-600 text-white shadow-sm'
                : 'bg-white text-red-700 border border-red-200 hover:bg-red-50'
            }`}
          >
            Resurfacing
          </button>
          <button
            type="button"
            onClick={() => setFilterAction('repair')}
            className={`px-3 py-1.5 rounded-lg font-medium transition-all ${
              filterAction === 'repair'
                ? 'bg-amber-600 text-white shadow-sm'
                : 'bg-white text-amber-700 border border-amber-200 hover:bg-amber-50'
            }`}
          >
            Repair
          </button>
          <button
            type="button"
            onClick={() => setFilterAction('none')}
            className={`px-3 py-1.5 rounded-lg font-medium transition-all ${
              filterAction === 'none'
                ? 'bg-emerald-600 text-white shadow-sm'
                : 'bg-white text-emerald-700 border border-emerald-200 hover:bg-emerald-50'
            }`}
          >
            Optimal
          </button>
        </div>
      </div>

      {/* Table */}
      <div className="overflow-x-auto rounded-xl border border-[#9B4A57]/15 bg-white/90 shadow-sm">
        <table className="w-full text-left text-xs border-collapse">
          <thead>
            <tr className="bg-[#FBF7F6] border-b border-[#9B4A57]/10 text-[#7C6A6D] font-heading font-semibold uppercase tracking-wider text-[11px]">
              <th className="py-3 px-4">Segment</th>
              <th className="py-3 px-4">Frame Window</th>
              <th className="py-3 px-4">Defects Detected</th>
              <th className="py-3 px-4">PCI Score</th>
              <th className="py-3 px-4">Condition</th>
              <th className="py-3 px-4">DQN Recommendation</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-[#9B4A57]/10">
            {filteredSegments.length === 0 ? (
              <tr>
                <td colSpan={6} className="py-8 text-center text-[#7C6A6D]">
                  No segments match your current search or filter.
                </td>
              </tr>
            ) : (
              filteredSegments.map((segment) => {
                let badgeClass = 'bg-emerald-50 text-emerald-700 border-emerald-200';
                if (segment.pci_score < 40) {
                  badgeClass = 'bg-rose-50 text-rose-700 border-rose-200';
                } else if (segment.pci_score < 70) {
                  badgeClass = 'bg-amber-50 text-amber-700 border-amber-200';
                }

                return (
                  <tr key={segment.segment_id} className="hover:bg-[#FBF7F6]/60 transition-colors">
                    <td className="py-3 px-4 font-heading font-bold text-[#2A1418]">
                      #{segment.segment_id}
                    </td>
                    <td className="py-3 px-4 text-[#7C6A6D] font-mono">
                      {segment.frame_start} - {segment.frame_end}
                    </td>
                    <td className="py-3 px-4">
                      <div className="flex items-center gap-1.5 flex-wrap">
                        {segment.defects.potholes > 0 && (
                          <span className="px-1.5 py-0.5 rounded bg-red-100 text-red-800 font-medium text-[10px]">
                            {segment.defects.potholes} Potholes
                          </span>
                        )}
                        {segment.defects.cracks > 0 && (
                          <span className="px-1.5 py-0.5 rounded bg-orange-100 text-orange-800 font-medium text-[10px]">
                            {segment.defects.cracks} Cracks
                          </span>
                        )}
                        {segment.defects.patches > 0 && (
                          <span className="px-1.5 py-0.5 rounded bg-amber-100 text-amber-800 font-medium text-[10px]">
                            {segment.defects.patches} Patches
                          </span>
                        )}
                        {segment.defects.digs > 0 && (
                          <span className="px-1.5 py-0.5 rounded bg-[#9B4A57]/15 text-[#9B4A57] font-medium text-[10px]">
                            {segment.defects.digs} Digs
                          </span>
                        )}
                        {segment.defects.digs === 0 &&
                          segment.defects.patches === 0 &&
                          segment.defects.cracks === 0 &&
                          segment.defects.potholes === 0 && (
                            <span className="text-[#7C6A6D] italic text-[11px]">Clean</span>
                          )}
                      </div>
                    </td>
                    <td className="py-3 px-4">
                      <div className="flex items-center gap-2">
                        <span className="font-heading font-bold text-[#2A1418]">
                          {segment.pci_score.toFixed(1)}
                        </span>
                        <div className="w-16 h-1.5 bg-[#F1EAE8] rounded-full overflow-hidden hidden sm:block">
                          <div
                            className={`h-full rounded-full ${
                              segment.pci_score < 40
                                ? 'bg-red-500'
                                : segment.pci_score < 70
                                ? 'bg-amber-500'
                                : 'bg-emerald-500'
                            }`}
                            style={{ width: `${segment.pci_score}%` }}
                          />
                        </div>
                      </div>
                    </td>
                    <td className="py-3 px-4">
                      <span className={`px-2 py-0.5 rounded-full text-[10px] font-semibold border ${badgeClass}`}>
                        {segment.condition_category}
                      </span>
                    </td>
                    <td className="py-3 px-4">
                      <span
                        className={`font-medium ${
                          segment.rl_recommended_action === 2
                            ? 'text-red-700 font-semibold'
                            : segment.rl_recommended_action === 1
                            ? 'text-amber-700'
                            : 'text-emerald-700'
                        }`}
                      >
                        {segment.rl_action_name}
                      </span>
                    </td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
};
