"use client";

import * as React from "react";
import * as TooltipPrimitive from "@radix-ui/react-tooltip";
import * as PopoverPrimitive from "@radix-ui/react-popover";
import * as DialogPrimitive from "@radix-ui/react-dialog";
import {
  Plus,
  SlidersHorizontal,
  ArrowUp,
  X,
  Globe,
  Pencil,
  Paintbrush,
  Telescope,
  Lightbulb,
  Mic,
} from "lucide-react";

// --- Utility ---
type ClassValue = string | number | boolean | null | undefined;
function cn(...inputs: ClassValue[]): string {
  return inputs.filter(Boolean).join(" ");
}

// --- Radix Primitives ---
const TooltipProvider = TooltipPrimitive.Provider;
const Tooltip = TooltipPrimitive.Root;
const TooltipTrigger = TooltipPrimitive.Trigger;
const TooltipContent = React.forwardRef<
  React.ElementRef<typeof TooltipPrimitive.Content>,
  React.ComponentPropsWithoutRef<typeof TooltipPrimitive.Content> & { showArrow?: boolean }
>(({ className, sideOffset = 4, showArrow = false, ...props }, ref) => (
  <TooltipPrimitive.Portal>
    <TooltipPrimitive.Content
      ref={ref}
      sideOffset={sideOffset}
      className={cn(
        "relative z-50 max-w-[280px] rounded-lg bg-[var(--color-gray-950)] text-xs px-2 py-1.5 text-white border border-[var(--color-gray-900)] shadow-xl",
        className
      )}
      {...props}
    >
      {props.children}
      {showArrow && <TooltipPrimitive.Arrow className="-my-px fill-[var(--color-gray-950)]" />}
    </TooltipPrimitive.Content>
  </TooltipPrimitive.Portal>
));
TooltipContent.displayName = TooltipPrimitive.Content.displayName;

const Popover = PopoverPrimitive.Root;
const PopoverTrigger = PopoverPrimitive.Trigger;
const PopoverContent = React.forwardRef<
  React.ElementRef<typeof PopoverPrimitive.Content>,
  React.ComponentPropsWithoutRef<typeof PopoverPrimitive.Content>
>(({ className, align = "center", sideOffset = 4, ...props }, ref) => (
  <PopoverPrimitive.Portal>
    <PopoverPrimitive.Content
      ref={ref}
      align={align}
      sideOffset={sideOffset}
      className={cn(
        "z-50 w-64 rounded-xl bg-[var(--color-gray-850)] p-2 text-white shadow-xl outline-none border border-[var(--color-gray-800)]",
        className
      )}
      {...props}
    />
  </PopoverPrimitive.Portal>
));
PopoverContent.displayName = PopoverPrimitive.Content.displayName;

const Dialog = DialogPrimitive.Root;
const DialogPortal = DialogPrimitive.Portal;
const DialogOverlay = React.forwardRef<
  React.ElementRef<typeof DialogPrimitive.Overlay>,
  React.ComponentPropsWithoutRef<typeof DialogPrimitive.Overlay>
>(({ className, ...props }, ref) => (
  <DialogPrimitive.Overlay
    ref={ref}
    className={cn("fixed inset-0 z-50 bg-black/60 backdrop-blur-sm", className)}
    {...props}
  />
));
DialogOverlay.displayName = DialogPrimitive.Overlay.displayName;

const DialogContent = React.forwardRef<
  React.ElementRef<typeof DialogPrimitive.Content>,
  React.ComponentPropsWithoutRef<typeof DialogPrimitive.Content>
>(({ className, children, ...props }, ref) => (
  <DialogPortal>
    <DialogOverlay />
    <DialogPrimitive.Content
      ref={ref}
      className={cn(
        "fixed left-[50%] top-[50%] z-50 grid w-full max-w-[90vw] md:max-w-[800px]",
        "translate-x-[-50%] translate-y-[-50%] gap-4 border-none bg-transparent p-0 shadow-none",
        className
      )}
      {...props}
    >
      <div className="relative bg-[var(--color-gray-850)] rounded-[28px] overflow-hidden shadow-2xl p-1">
        {children}
        <DialogPrimitive.Close className="absolute right-3 top-3 z-10 rounded-full bg-[var(--color-gray-850)] p-1 hover:bg-[var(--color-gray-800)] transition-all cursor-pointer">
          <X className="h-5 w-5 text-[var(--color-gray-200)]" />
          <span className="sr-only">Close</span>
        </DialogPrimitive.Close>
      </div>
    </DialogPrimitive.Content>
  </DialogPortal>
));
DialogContent.displayName = DialogPrimitive.Content.displayName;

// --- Tools List ---
const toolsList = [
  { id: "createImage", name: "Create an image", shortName: "Image", icon: Paintbrush },
  { id: "searchWeb", name: "Search the web", shortName: "Search", icon: Globe },
  { id: "writeCode", name: "Write or code", shortName: "Write", icon: Pencil },
  { id: "deepResearch", name: "Run deep research", shortName: "Deep Search", icon: Telescope, extra: "5 left" },
  { id: "thinkLonger", name: "Think for longer", shortName: "Think", icon: Lightbulb },
];

// --- Props ---
interface PromptBoxProps extends React.TextareaHTMLAttributes<HTMLTextAreaElement> {
  onSendMessage?: (message: string) => void;
}

// --- PromptBox Component ---
export const PromptBox = React.forwardRef<HTMLTextAreaElement, PromptBoxProps>(
  ({ className, onSendMessage, ...props }, ref) => {
    const internalTextareaRef = React.useRef<HTMLTextAreaElement>(null);
    const fileInputRef = React.useRef<HTMLInputElement>(null);
    const [value, setValue] = React.useState("");
    const [imagePreview, setImagePreview] = React.useState<string | null>(null);
    const [selectedTool, setSelectedTool] = React.useState<string | null>(null);
    const [isPopoverOpen, setIsPopoverOpen] = React.useState(false);
    const [isImageDialogOpen, setIsImageDialogOpen] = React.useState(false);

    React.useImperativeHandle(ref, () => internalTextareaRef.current!, []);

    React.useLayoutEffect(() => {
      const textarea = internalTextareaRef.current;
      if (textarea) {
        textarea.style.height = "auto";
        const newHeight = Math.min(textarea.scrollHeight, 200);
        textarea.style.height = `${newHeight}px`;
      }
    }, [value]);

    const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
      setValue(e.target.value);
      if (props.onChange) props.onChange(e);
    };

    const handleSend = () => {
      if (value.trim() && onSendMessage) {
        onSendMessage(value.trim());
        setValue("");
      }
    };

    const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleSend();
      }
    };

    const handlePlusClick = () => { fileInputRef.current?.click(); };

    const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0];
      if (file && file.type.startsWith("image/")) {
        const reader = new FileReader();
        reader.onloadend = () => { setImagePreview(reader.result as string); };
        reader.readAsDataURL(file);
      }
      event.target.value = "";
    };

    const handleRemoveImage = (e: React.MouseEvent<HTMLButtonElement>) => {
      e.stopPropagation();
      setImagePreview(null);
      if (fileInputRef.current) fileInputRef.current.value = "";
    };

    const hasValue = value.trim().length > 0 || imagePreview;
    const activeTool = selectedTool ? toolsList.find((t) => t.id === selectedTool) : null;
    const ActiveToolIcon = activeTool?.icon;

    return (
      <div
        className={cn(
          "flex flex-col rounded-3xl p-1.5 shadow-sm transition-colors",
          "bg-[var(--color-gray-850)] border border-[var(--color-gray-800)] cursor-text",
          className
        )}
        onClick={() => internalTextareaRef.current?.focus()}
      >
        <input type="file" ref={fileInputRef} onChange={handleFileChange} className="hidden" accept="image/*" />

        {/* Image Preview */}
        {imagePreview && (
          <Dialog open={isImageDialogOpen} onOpenChange={setIsImageDialogOpen}>
            <div className="relative mb-1 w-fit rounded-2xl px-1 pt-1">
              <button type="button" className="transition-transform hover:scale-[1.02]" onClick={() => setIsImageDialogOpen(true)}>
                <img src={imagePreview} alt="Image preview" className="h-14 w-14 rounded-2xl object-cover" />
              </button>
              <button onClick={handleRemoveImage} className="absolute right-2 top-2 z-10 flex h-4 w-4 items-center justify-center rounded-full bg-[var(--color-gray-850)] text-white transition-colors hover:bg-[var(--color-gray-800)]" aria-label="Remove image">
                <X className="h-3 w-3" />
              </button>
            </div>
            <DialogContent>
              <img src={imagePreview} alt="Full size preview" className="w-full max-h-[95vh] object-contain rounded-3xl" />
            </DialogContent>
          </Dialog>
        )}

        {/* Textarea */}
        <textarea
          id="chat-input"
          ref={internalTextareaRef}
          rows={1}
          value={value}
          onChange={handleInputChange}
          onKeyDown={handleKeyDown}
          placeholder={props.placeholder || "How can I help you today?"}
          className="w-full resize-none border-0 bg-transparent px-3 py-2.5 text-white placeholder:text-[var(--color-gray-400)] focus:ring-0 focus-visible:outline-none min-h-[44px] text-[15px]"
          {...props}
        />

        {/* Bottom Toolbar */}
        <div className="p-1 pt-0">
          <TooltipProvider delayDuration={100}>
            <div className="flex items-center gap-1.5">
              {/* Attach */}
              <Tooltip>
                <TooltipTrigger asChild>
                  <button type="button" onClick={handlePlusClick} className="flex h-8 w-8 items-center justify-center rounded-full text-[var(--color-gray-400)] transition-colors hover:bg-[var(--color-gray-800)] hover:text-white focus-visible:outline-none">
                    <Plus className="h-5 w-5" />
                    <span className="sr-only">Attach</span>
                  </button>
                </TooltipTrigger>
                <TooltipContent side="top" showArrow><p>Attach</p></TooltipContent>
              </Tooltip>

              {/* Tools */}
              <Popover open={isPopoverOpen} onOpenChange={setIsPopoverOpen}>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <PopoverTrigger asChild>
                      <button type="button" className="flex h-8 items-center gap-1.5 rounded-full px-2 text-sm text-[var(--color-gray-400)] transition-colors hover:bg-[var(--color-gray-800)] hover:text-white focus-visible:outline-none">
                        <SlidersHorizontal className="h-4 w-4" />
                        {!selectedTool && <span className="text-xs">Tools</span>}
                      </button>
                    </PopoverTrigger>
                  </TooltipTrigger>
                  <TooltipContent side="top" showArrow><p>Explore Tools</p></TooltipContent>
                </Tooltip>
                <PopoverContent side="top" align="start">
                  <div className="flex flex-col gap-0.5">
                    {toolsList.map((tool) => (
                      <button
                        key={tool.id}
                        onClick={() => { setSelectedTool(tool.id); setIsPopoverOpen(false); }}
                        className="flex w-full items-center gap-2 rounded-lg p-2 text-left text-sm hover:bg-[var(--color-gray-800)] transition"
                      >
                        <tool.icon className="h-4 w-4" />
                        <span>{tool.name}</span>
                        {tool.extra && <span className="ml-auto text-xs text-[var(--color-gray-500)]">{tool.extra}</span>}
                      </button>
                    ))}
                  </div>
                </PopoverContent>
              </Popover>

              {/* Active Tool Badge */}
              {activeTool && (
                <>
                  <div className="h-4 w-px bg-[var(--color-gray-700)]" />
                  <button
                    onClick={() => setSelectedTool(null)}
                    className="flex h-7 items-center gap-1.5 rounded-full px-2 text-xs hover:bg-[var(--color-gray-800)] text-blue-400 transition"
                  >
                    {ActiveToolIcon && <ActiveToolIcon className="h-3.5 w-3.5" />}
                    {activeTool.shortName}
                    <X className="h-3 w-3" />
                  </button>
                </>
              )}

              {/* Right-aligned */}
              <div className="ml-auto flex items-center gap-1.5">
                <Tooltip>
                  <TooltipTrigger asChild>
                    <button type="button" className="flex h-8 w-8 items-center justify-center rounded-full text-[var(--color-gray-400)] transition-colors hover:bg-[var(--color-gray-800)] hover:text-white focus-visible:outline-none">
                      <Mic className="h-4.5 w-4.5" />
                      <span className="sr-only">Voice</span>
                    </button>
                  </TooltipTrigger>
                  <TooltipContent side="top" showArrow><p>Voice</p></TooltipContent>
                </Tooltip>

                <Tooltip>
                  <TooltipTrigger asChild>
                    <button
                      type="button"
                      onClick={handleSend}
                      disabled={!hasValue}
                      className="flex h-8 w-8 items-center justify-center rounded-full text-sm font-medium transition-colors focus-visible:outline-none disabled:pointer-events-none bg-white text-black hover:bg-[var(--color-gray-200)] disabled:bg-[var(--color-gray-700)] disabled:text-[var(--color-gray-500)]"
                    >
                      <ArrowUp className="h-4.5 w-4.5" strokeWidth={2.5} />
                      <span className="sr-only">Send</span>
                    </button>
                  </TooltipTrigger>
                  <TooltipContent side="top" showArrow><p>Send</p></TooltipContent>
                </Tooltip>
              </div>
            </div>
          </TooltipProvider>
        </div>
      </div>
    );
  }
);
PromptBox.displayName = "PromptBox";
