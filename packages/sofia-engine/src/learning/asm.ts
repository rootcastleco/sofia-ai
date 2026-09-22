/**
 * Assembly Virtual Machine & Bytecode Engine for Neural Self-Training.
 * Zero-dependency, Float64Array-based VM for deterministic on-device training.
 * Developed by Rootcastle Engineering & Innovation (https://rootcastle.com/)
 */

export enum Register {
  R0 = 0,
  R1 = 1,
  R2 = 2,
  R3 = 3,
  R4 = 4,
  R5 = 5,
  R6 = 6,
  R7 = 7,
  ACC = 8,
  LR = 9,
  ERR = 10,
  PC = 11,
  SP = 12,
}

export enum OpCode {
  NOP = 0x00,
  LOAD_CONST = 0x01,
  LOAD_MEM = 0x02,
  STORE_MEM = 0x03,
  MOV = 0x04,

  // Arithmetic
  ADD = 0x10,
  SUB = 0x11,
  MUL = 0x12,
  DIV = 0x13,
  FMA = 0x14,

  // Vectorized Neural Primitives
  VEC_DOT = 0x20,
  VEC_FMA = 0x21,
  VEC_SUB = 0x22,

  // Activations & Derivatives
  ACT_RELU = 0x30,
  GRAD_RELU = 0x31,
  ACT_SIGMOID = 0x32,
  GRAD_SIGMOID = 0x33,

  // Self-Training & Optimization
  UPDATE_SGD = 0x40,
  COMPUTE_MSE = 0x41,

  // Control Flow
  JMP = 0x50,
  JZ = 0x51,
  HALT = 0xff,
}

export enum VMStatus {
  HALTED = "HALTED",
  RUNNING = "RUNNING",
  CYCLE_LIMIT = "CYCLE_LIMIT",
  FAULT = "FAULT"
}

export enum VMFault {
  NONE = "NONE",
  MEM_OUT_OF_BOUNDS = "MEM_OUT_OF_BOUNDS",
  REG_OUT_OF_BOUNDS = "REG_OUT_OF_BOUNDS",
  INVALID_OPCODE = "INVALID_OPCODE",
  INVALID_PROGRAM = "INVALID_PROGRAM",
  CYCLE_LIMIT = "CYCLE_LIMIT"
}

export interface VMExecutionResult {
  status: VMStatus;
  fault: VMFault;
  cycles: number;
  pc: number;
}

export interface AsmInstruction {
  opcode: OpCode;
  arg1?: number;
  arg2?: number;
  arg3?: number;
  imm?: number;
}

export class SofiaAsmVM {
  public memory: Float64Array;
  public registers: Float64Array;
  public pc: number = 0;
  public halted: boolean = false;

  constructor(memorySize: number = 65536) {
    this.memory = new Float64Array(memorySize);
    this.registers = new Float64Array(16);
  }

  public getReg(reg: number): number {
    return this.registers[reg] ?? 0.0;
  }

  public setReg(reg: number, val: number): void {
    if (reg >= 0 && reg < this.registers.length) {
      this.registers[reg] = val;
    }
  }

  public getMem(addr: number): number {
    return this.memory[addr] ?? 0.0;
  }

  public setMem(addr: number, val: number): void {
    if (addr >= 0 && addr < this.memory.length) {
      this.memory[addr] = val;
    }
  }

  public reset(): void {
    this.registers.fill(0);
    this.pc = 0;
    this.halted = false;
  }

  public step(inst: AsmInstruction): void {
    const { opcode, arg1 = 0, arg2 = 0, arg3 = 0, imm = 0.0 } = inst;

    switch (opcode) {
      case OpCode.NOP:
        break;

      case OpCode.LOAD_CONST:
        this.setReg(arg1, imm);
        break;

      case OpCode.LOAD_MEM: {
        const addr = Math.floor(this.getReg(arg2));
        this.setReg(arg1, this.getMem(addr));
        break;
      }

      case OpCode.STORE_MEM: {
        const addr = Math.floor(this.getReg(arg1));
        this.setMem(addr, this.getReg(arg2));
        break;
      }

      case OpCode.MOV:
        this.setReg(arg1, this.getReg(arg2));
        break;

      case OpCode.ADD: {
        const res = this.getReg(arg1) + this.getReg(arg2);
        this.setReg(arg1, res);
        this.setReg(Register.ACC, res);
        break;
      }

      case OpCode.SUB: {
        const res = this.getReg(arg1) - this.getReg(arg2);
        this.setReg(arg1, res);
        this.setReg(Register.ACC, res);
        break;
      }

      case OpCode.MUL: {
        const res = this.getReg(arg1) * this.getReg(arg2);
        this.setReg(arg1, res);
        this.setReg(Register.ACC, res);
        break;
      }

      case OpCode.DIV: {
        const d = this.getReg(arg2);
        const res = d !== 0.0 ? this.getReg(arg1) / d : 0.0;
        this.setReg(arg1, res);
        this.setReg(Register.ACC, res);
        break;
      }

      case OpCode.FMA: {
        const res = this.getReg(arg1) + this.getReg(arg2) * this.getReg(arg3);
        this.setReg(arg1, res);
        this.setReg(Register.ACC, res);
        break;
      }

      case OpCode.VEC_DOT: {
        const addrA = Math.floor(this.getReg(arg1));
        const addrB = Math.floor(this.getReg(arg2));
        const length = Math.floor(this.getReg(arg3));
        let sum = 0.0;
        for (let i = 0; i < length; i++) {
          sum += this.getMem(addrA + i) * this.getMem(addrB + i);
        }
        this.setReg(Register.ACC, sum);
        break;
      }

      case OpCode.VEC_FMA: {
        const addrDest = Math.floor(this.getReg(arg1));
        const addrSrc = Math.floor(this.getReg(arg2));
        const scalar = this.getReg(arg3);
        const length = Math.floor(imm);
        for (let i = 0; i < length; i++) {
          this.setMem(addrDest + i, this.getMem(addrDest + i) + this.getMem(addrSrc + i) * scalar);
        }
        break;
      }

      case OpCode.VEC_SUB: {
        const addrDest = Math.floor(this.getReg(arg1));
        const addrA = Math.floor(this.getReg(arg2));
        const addrB = Math.floor(this.getReg(arg3));
        const length = Math.floor(imm);
        for (let i = 0; i < length; i++) {
          this.setMem(addrDest + i, this.getMem(addrA + i) - this.getMem(addrB + i));
        }
        break;
      }

      case OpCode.ACT_RELU: {
        const addr = Math.floor(this.getReg(arg1));
        const length = Math.floor(imm);
        for (let i = 0; i < length; i++) {
          const val = this.getMem(addr + i);
          this.setMem(addr + i, val > 0.0 ? val : 0.0);
        }
        break;
      }

      case OpCode.UPDATE_SGD: {
        const addrW = Math.floor(this.getReg(arg1));
        const addrG = Math.floor(this.getReg(arg2));
        const lr = this.getReg(arg3);
        const length = Math.floor(imm);
        for (let i = 0; i < length; i++) {
          this.setMem(addrW + i, this.getMem(addrW + i) - lr * this.getMem(addrG + i));
        }
        break;
      }

      case OpCode.COMPUTE_MSE: {
        const addrP = Math.floor(this.getReg(arg1));
        const addrT = Math.floor(this.getReg(arg2));
        const length = Math.floor(imm);
        let mse = 0.0;
        for (let i = 0; i < length; i++) {
          const diff = this.getMem(addrP + i) - this.getMem(addrT + i);
          mse += diff * diff;
        }
        this.setReg(Register.ERR, length > 0 ? mse / length : 0.0);
        break;
      }

      case OpCode.JMP: {
        this.pc = Math.floor(imm);
        return;
      }

      case OpCode.JZ: {
        if (this.getReg(arg1) === 0.0) {
          this.pc = Math.floor(imm);
          return;
        }
        break;
      }

      case OpCode.HALT:
        this.halted = true;
        break;

      default:
        break;
    }

    this.pc++;
  }

  public run(program: AsmInstruction[], maxCycles: number = 100000): VMExecutionResult {
    this.halted = false;
    let cycles = 0;
    if (!program || program.length === 0) {
      return {
        status: VMStatus.FAULT,
        fault: VMFault.INVALID_PROGRAM,
        cycles: 0,
        pc: this.pc
      };
    }
    while (!this.halted && this.pc < program.length && cycles < maxCycles) {
      const inst = program[this.pc];
      if (!inst) {
        return {
          status: VMStatus.FAULT,
          fault: VMFault.INVALID_PROGRAM,
          cycles,
          pc: this.pc
        };
      }
      this.step(inst);
      cycles++;
    }
    if (cycles >= maxCycles && !this.halted) {
      return {
        status: VMStatus.CYCLE_LIMIT,
        fault: VMFault.CYCLE_LIMIT,
        cycles,
        pc: this.pc
      };
    }
    return {
      status: this.halted ? VMStatus.HALTED : VMStatus.RUNNING,
      fault: VMFault.NONE,
      cycles,
      pc: this.pc
    };
  }
}

export class AssemblyNeuralNetwork {
  public inputDim: number;
  public hiddenDim: number;
  public outputDim: number;
  public learningRate: number;
  public vm: SofiaAsmVM;

  public addrIn: number;
  public addrH: number;
  public addrOut: number;
  public addrTarget: number;
  public addrW1: number;
  public addrB1: number;
  public addrW2: number;
  public addrB2: number;
  public addrGradOut: number;
  public addrGradW2: number;

  constructor(
    inputDim: number,
    hiddenDim: number,
    outputDim: number,
    learningRate: number = 0.01
  ) {
    this.inputDim = inputDim;
    this.hiddenDim = hiddenDim;
    this.outputDim = outputDim;
    this.learningRate = learningRate;
    this.vm = new SofiaAsmVM(65536);

    this.addrIn = 0;
    this.addrH = this.addrIn + inputDim;
    this.addrOut = this.addrH + hiddenDim;
    this.addrTarget = this.addrOut + outputDim;
    this.addrW1 = this.addrTarget + outputDim;
    this.addrB1 = this.addrW1 + inputDim * hiddenDim;
    this.addrW2 = this.addrB1 + hiddenDim;
    this.addrB2 = this.addrW2 + hiddenDim * outputDim;
    this.addrGradOut = this.addrB2 + outputDim;
    this.addrGradW2 = this.addrGradOut + outputDim;

    // Xavier uniform initialization
    const limit1 = Math.sqrt(6.0 / (inputDim + hiddenDim));
    for (let i = 0; i < inputDim * hiddenDim; i++) {
      this.vm.setMem(this.addrW1 + i, (Math.random() * 2 - 1) * limit1);
    }

    const limit2 = Math.sqrt(6.0 / (hiddenDim + outputDim));
    for (let i = 0; i < hiddenDim * outputDim; i++) {
      this.vm.setMem(this.addrW2 + i, (Math.random() * 2 - 1) * limit2);
    }
  }

  public forward(x: number[] | Float64Array): number[] {
    for (let i = 0; i < Math.min(x.length, this.inputDim); i++) {
      const val = x[i];
      if (val !== undefined) {
        this.vm.setMem(this.addrIn + i, val);
      }
    }

    const program: AsmInstruction[] = [];

    // Hidden layer
    for (let j = 0; j < this.hiddenDim; j++) {
      const wRowAddr = this.addrW1 + j * this.inputDim;
      const bAddr = this.addrB1 + j;
      const hAddr = this.addrH + j;

      program.push(
        { opcode: OpCode.LOAD_CONST, arg1: Register.R0, imm: wRowAddr },
        { opcode: OpCode.LOAD_CONST, arg1: Register.R1, imm: this.addrIn },
        { opcode: OpCode.LOAD_CONST, arg1: Register.R2, imm: this.inputDim },
        { opcode: OpCode.VEC_DOT, arg1: Register.R0, arg2: Register.R1, arg3: Register.R2 },
        { opcode: OpCode.LOAD_CONST, arg1: Register.R3, imm: bAddr },
        { opcode: OpCode.LOAD_MEM, arg1: Register.R4, arg2: Register.R3 },
        { opcode: OpCode.ADD, arg1: Register.ACC, arg2: Register.R4 },
        { opcode: OpCode.LOAD_CONST, arg1: Register.R3, imm: hAddr },
        { opcode: OpCode.STORE_MEM, arg1: Register.R3, arg2: Register.ACC }
      );
    }

    // ReLU on hidden layer
    program.push(
      { opcode: OpCode.LOAD_CONST, arg1: Register.R0, imm: this.addrH },
      { opcode: OpCode.ACT_RELU, arg1: Register.R0, imm: this.hiddenDim }
    );

    // Output layer
    for (let k = 0; k < this.outputDim; k++) {
      const wRowAddr = this.addrW2 + k * this.hiddenDim;
      const bAddr = this.addrB2 + k;
      const outAddr = this.addrOut + k;

      program.push(
        { opcode: OpCode.LOAD_CONST, arg1: Register.R0, imm: wRowAddr },
        { opcode: OpCode.LOAD_CONST, arg1: Register.R1, imm: this.addrH },
        { opcode: OpCode.LOAD_CONST, arg1: Register.R2, imm: this.hiddenDim },
        { opcode: OpCode.VEC_DOT, arg1: Register.R0, arg2: Register.R1, arg3: Register.R2 },
        { opcode: OpCode.LOAD_CONST, arg1: Register.R3, imm: bAddr },
        { opcode: OpCode.LOAD_MEM, arg1: Register.R4, arg2: Register.R3 },
        { opcode: OpCode.ADD, arg1: Register.ACC, arg2: Register.R4 },
        { opcode: OpCode.LOAD_CONST, arg1: Register.R3, imm: outAddr },
        { opcode: OpCode.STORE_MEM, arg1: Register.R3, arg2: Register.ACC }
      );
    }

    program.push({ opcode: OpCode.HALT });
    this.vm.pc = 0;
    this.vm.run(program);

    const result: number[] = [];
    for (let i = 0; i < this.outputDim; i++) {
      result.push(this.vm.getMem(this.addrOut + i));
    }
    return result;
  }

  public trainStep(x: number[] | Float64Array, y: number[] | Float64Array): number {
    this.forward(x);
    for (let i = 0; i < Math.min(y.length, this.outputDim); i++) {
      const val = y[i];
      if (val !== undefined) {
        this.vm.setMem(this.addrTarget + i, val);
      }
    }

    const program: AsmInstruction[] = [
      // Compute MSE loss & output error
      { opcode: OpCode.LOAD_CONST, arg1: Register.R0, imm: this.addrOut },
      { opcode: OpCode.LOAD_CONST, arg1: Register.R1, imm: this.addrTarget },
      { opcode: OpCode.COMPUTE_MSE, arg1: Register.R0, arg2: Register.R1, imm: this.outputDim },
      { opcode: OpCode.LOAD_CONST, arg1: Register.R0, imm: this.addrGradOut },
      { opcode: OpCode.LOAD_CONST, arg1: Register.R1, imm: this.addrOut },
      { opcode: OpCode.LOAD_CONST, arg1: Register.R2, imm: this.addrTarget },
      { opcode: OpCode.VEC_SUB, arg1: Register.R0, arg2: Register.R1, arg3: Register.R2, imm: this.outputDim },
    ];

    // Compute gradients for W2 and update via SGD
    for (let k = 0; k < this.outputDim; k++) {
      const gradOutK = this.addrGradOut + k;
      const w2RowGrad = this.addrGradW2 + k * this.hiddenDim;
      program.push(
        { opcode: OpCode.LOAD_CONST, arg1: Register.R0, imm: w2RowGrad },
        { opcode: OpCode.LOAD_CONST, arg1: Register.R1, imm: this.addrH },
        { opcode: OpCode.LOAD_CONST, arg1: Register.R2, imm: gradOutK },
        { opcode: OpCode.LOAD_MEM, arg1: Register.R3, arg2: Register.R2 },
        { opcode: OpCode.VEC_FMA, arg1: Register.R0, arg2: Register.R1, arg3: Register.R3, imm: this.hiddenDim }
      );
    }

    // SGD weight updates
    program.push(
      { opcode: OpCode.LOAD_CONST, arg1: Register.LR, imm: this.learningRate },
      { opcode: OpCode.LOAD_CONST, arg1: Register.R0, imm: this.addrW2 },
      { opcode: OpCode.LOAD_CONST, arg1: Register.R1, imm: this.addrGradW2 },
      { opcode: OpCode.UPDATE_SGD, arg1: Register.R0, arg2: Register.R1, arg3: Register.LR, imm: this.hiddenDim * this.outputDim },
      { opcode: OpCode.LOAD_CONST, arg1: Register.R0, imm: this.addrB2 },
      { opcode: OpCode.LOAD_CONST, arg1: Register.R1, imm: this.addrGradOut },
      { opcode: OpCode.UPDATE_SGD, arg1: Register.R0, arg2: Register.R1, arg3: Register.LR, imm: this.outputDim },
      { opcode: OpCode.HALT }
    );

    this.vm.pc = 0;
    this.vm.run(program);
    return this.vm.getReg(Register.ERR);
  }
}
