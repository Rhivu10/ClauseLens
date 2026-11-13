// Dropzone: drag and drop or click to choose files (react-dropzone). Adapted from
// the shadcn-style dropzone component to plain JSX and the ClauseLens colours.
//
//   <Dropzone accept={{ "application/pdf": [".pdf"] }} src={files} onDrop={setFiles} onError={...}>
//     <DropzoneEmptyState />     shown while no file is chosen
//     <DropzoneContent />        shown once one is (the file names)
//   </Dropzone>
import { createContext, useContext } from "react";
import { useDropzone } from "react-dropzone";
import { FileText, UploadIcon } from "lucide-react";

const DropzoneContext = createContext(undefined);

export const renderBytes = (bytes) => {
  const units = ["B", "KB", "MB", "GB"];
  let size = bytes;
  let i = 0;
  while (size >= 1024 && i < units.length - 1) {
    size /= 1024;
    i++;
  }
  return `${size.toFixed(i === 0 ? 0 : 2)} ${units[i]}`;
};

export function Dropzone({ accept, maxFiles = 1, maxSize, minSize, onDrop, onError, disabled, src, className = "", children, ...props }) {
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept,
    maxFiles,
    maxSize,
    minSize,
    disabled,
    onError,
    onDrop: (accepted, rejections, event) => {
      if (rejections.length > 0) {
        onError?.(new Error(rejections[0]?.errors[0]?.message || "That file can't be used."));
        return;
      }
      onDrop?.(accepted, rejections, event);
    },
    ...props,
  });

  return (
    <DropzoneContext.Provider key={JSON.stringify(src?.map((f) => f.name))} value={{ src, accept, maxSize, minSize, maxFiles }}>
      <button
        type="button"
        disabled={disabled}
        className={`relative w-full h-auto flex flex-col items-center justify-center text-center overflow-hidden p-8 rounded-2xl border transition-colors
          bg-(--color-surface) hover:bg-(--color-surface-2)/40 disabled:opacity-50 disabled:pointer-events-none
          focus:outline-none focus-visible:ring-2 focus-visible:ring-(--color-brass)
          ${isDragActive ? "border-(--color-brass) ring-1 ring-(--color-brass) bg-(--color-brass)/5" : "border-(--color-line)"} ${className}`}
        {...getRootProps()}
      >
        <input {...getInputProps()} disabled={disabled} />
        {children}
      </button>
    </DropzoneContext.Provider>
  );
}

const useDropzoneContext = () => {
  const context = useContext(DropzoneContext);
  if (!context) throw new Error("useDropzoneContext must be used within a Dropzone");
  return context;
};

const MAX_LABEL_ITEMS = 3;
const list = (names) => new Intl.ListFormat("en").format(names);

const IconTile = ({ children }) => (
  <div className="flex size-10 items-center justify-center rounded-lg bg-(--color-surface-2) text-(--color-ink-dim)">{children}</div>
);

export function DropzoneContent({ children }) {
  const { src } = useDropzoneContext();
  if (!src?.length) return null;
  if (children) return children;
  return (
    <>
      <IconTile>
        <FileText size={18} />
      </IconTile>
      <p className="mt-3 mb-1 w-full truncate font-medium text-[15px]">
        {src.length > MAX_LABEL_ITEMS
          ? `${list(src.slice(0, MAX_LABEL_ITEMS).map((f) => f.name))} and ${src.length - MAX_LABEL_ITEMS} more`
          : list(src.map((f) => f.name))}
      </p>
      <p className="w-full font-mono text-[11px] text-(--color-ink-faint)">
        {renderBytes(src.reduce((n, f) => n + f.size, 0))} · ready to compare
      </p>
      <p className="mt-1 w-full text-[13px] text-(--color-ink-faint)">Drag and drop or click to replace</p>
    </>
  );
}

export function DropzoneEmptyState({ children }) {
  const { src, accept, maxSize, minSize, maxFiles } = useDropzoneContext();
  if (src?.length) return null;
  if (children) return children;

  let caption = "";
  if (accept) caption += `Accepts ${list(Object.values(accept).flat().length ? Object.values(accept).flat() : Object.keys(accept))}`;
  if (minSize && maxSize) caption += ` between ${renderBytes(minSize)} and ${renderBytes(maxSize)}`;
  else if (minSize) caption += ` at least ${renderBytes(minSize)}`;
  else if (maxSize) caption += ` up to ${renderBytes(maxSize)}`;

  return (
    <>
      <IconTile>
        <UploadIcon size={18} />
      </IconTile>
      <p className="mt-3 mb-1 w-full truncate font-medium text-[15px]">Upload {maxFiles === 1 ? "a file" : "files"}</p>
      <p className="w-full truncate text-[13px] text-(--color-ink-faint)">Drag and drop or click to upload</p>
      {caption && <p className="text-[13px] text-(--color-ink-faint)">{caption.trim()}.</p>}
    </>
  );
}
