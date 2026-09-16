import pc from 'picocolors';
import { format } from './format.js';

export class TokenStreamRenderer {
  private inReasoning = false;
  private hasContentStarted = false;

  public handleReasoningDelta(delta: string) {
    if (!this.inReasoning) {
      this.inReasoning = true;
      process.stderr.write(format.reasoningHeader());
    }
    process.stderr.write(pc.dim(delta));
  }

  public handleContentDelta(delta: string) {
    if (this.inReasoning) {
      this.inReasoning = false;
      process.stderr.write(format.reasoningFooter());
    }
    if (!this.hasContentStarted) {
      this.hasContentStarted = true;
    }
    process.stdout.write(delta);
  }

  public finish() {
    if (this.inReasoning) {
      this.inReasoning = false;
      process.stderr.write(format.reasoningFooter());
    }
    if (this.hasContentStarted) {
      process.stdout.write('\n');
    }
    this.hasContentStarted = false;
  }
}
