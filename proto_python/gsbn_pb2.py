# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: gsbn.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='gsbn.proto',
  package='gsbn',
  serialized_pb=_b('\n\ngsbn.proto\x12\x04gsbn\"v\n\x0bSolverParam\x12!\n\tgen_param\x18\x01 \x02(\x0b\x32\x0e.gsbn.GenParam\x12!\n\tnet_param\x18\x02 \x02(\x0b\x32\x0e.gsbn.NetParam\x12!\n\trec_param\x18\x03 \x02(\x0b\x32\x0e.gsbn.RecParam\"N\n\x08GenParam\x12\x11\n\tstim_file\x18\x01 \x01(\t\x12\n\n\x02\x64t\x18\x02 \x02(\x02\x12#\n\nmode_param\x18\x03 \x03(\x0b\x32\x0f.gsbn.ModeParam\"\x85\x01\n\tModeParam\x12\x12\n\nbegin_time\x18\x01 \x02(\x02\x12\x10\n\x08\x65nd_time\x18\x02 \x02(\x02\x12\x0e\n\x03prn\x18\x03 \x01(\x02:\x01\x31\x12\x14\n\tgain_mask\x18\x04 \x01(\r:\x01\x30\x12\x15\n\nplasticity\x18\x05 \x01(\r:\x01\x31\x12\x15\n\nstim_index\x18\x06 \x01(\r:\x01\x30\"e\n\x08NetParam\x12!\n\tpop_param\x18\x01 \x03(\x0b\x32\x0e.gsbn.PopParam\x12#\n\nproj_param\x18\x02 \x03(\x0b\x32\x0f.gsbn.ProjParam\x12\x11\n\tprocedure\x18\x03 \x03(\t\"\xe9\x01\n\x08PopParam\x12\x0f\n\x07pop_num\x18\x01 \x02(\r\x12\x0f\n\x07hcu_num\x18\x02 \x02(\r\x12\x0f\n\x07mcu_num\x18\x03 \x02(\r\x12\x10\n\x08slot_num\x18\x04 \x02(\r\x12\x12\n\nfanout_num\x18\x05 \x02(\r\x12\x12\n\x04taum\x18\x06 \x01(\x02:\x04\x30.01\x12\x12\n\x07wtagain\x18\x07 \x01(\x02:\x01\x34\x12\x12\n\x05maxfq\x18\x08 \x01(\x02:\x03\x31\x30\x30\x12\x10\n\x05igain\x18\t \x01(\x02:\x01\x31\x12\x10\n\x05wgain\x18\n \x01(\x02:\x01\x31\x12\x11\n\x06lgbias\x18\x0b \x01(\x02:\x01\x30\x12\x11\n\x06snoise\x18\x0c \x01(\x02:\x01\x30\"\xb0\x01\n\tProjParam\x12\x0f\n\x07src_pop\x18\x01 \x02(\r\x12\x10\n\x08\x64\x65st_pop\x18\x02 \x02(\r\x12\r\n\x05tauzi\x18\x03 \x02(\x02\x12\r\n\x05tauzj\x18\x04 \x02(\x02\x12\x0c\n\x04taue\x18\x05 \x02(\x02\x12\x0c\n\x04taup\x18\x06 \x02(\x02\x12\x12\n\x05maxfq\x18\x07 \x01(\x02:\x03\x31\x30\x30\x12\x10\n\x05\x62gain\x18\x08 \x01(\x02:\x01\x30\x12\x10\n\x05wgain\x18\t \x01(\x02:\x01\x30\x12\x0e\n\x03pi0\x18\n \x01(\x02:\x01\x30\"~\n\x08RecParam\x12\x14\n\tdirectory\x18\x01 \x01(\t:\x01.\x12\x14\n\x06\x65nable\x18\x02 \x01(\x08:\x04true\x12\x11\n\x06offset\x18\x03 \x01(\r:\x01\x30\x12\x1a\n\x0fsnapshot_period\x18\x04 \x01(\x05:\x01\x30\x12\x17\n\x0cspike_period\x18\x05 \x01(\x05:\x01\x30\"\xd8\x01\n\x0bSolverState\x12\x11\n\ttimestamp\x18\x01 \x02(\x02\x12\x0b\n\x03prn\x18\x02 \x02(\x02\x12%\n\x0btable_state\x18\x03 \x03(\x0b\x32\x10.gsbn.TableState\x12*\n\x0evector_state_i\x18\x04 \x03(\x0b\x32\x12.gsbn.VectorStateI\x12*\n\x0evector_state_f\x18\x05 \x03(\x0b\x32\x12.gsbn.VectorStateF\x12*\n\x0evector_state_d\x18\x06 \x03(\x0b\x32\x12.gsbn.VectorStateD\"6\n\nTableState\x12\x0c\n\x04name\x18\x01 \x02(\t\x12\x0c\n\x04\x64\x65sc\x18\x02 \x02(\x0c\x12\x0c\n\x04\x64\x61ta\x18\x03 \x02(\x0c\"=\n\x0cVectorStateI\x12\x0c\n\x04name\x18\x01 \x02(\t\x12\r\n\x02ld\x18\x02 \x01(\r:\x01\x31\x12\x10\n\x04\x64\x61ta\x18\x03 \x03(\x05\x42\x02\x10\x01\"=\n\x0cVectorStateF\x12\x0c\n\x04name\x18\x01 \x02(\t\x12\r\n\x02ld\x18\x02 \x01(\r:\x01\x31\x12\x10\n\x04\x64\x61ta\x18\x03 \x03(\x02\x42\x02\x10\x01\"=\n\x0cVectorStateD\x12\x0c\n\x04name\x18\x01 \x02(\t\x12\r\n\x02ld\x18\x02 \x01(\r:\x01\x31\x12\x10\n\x04\x64\x61ta\x18\x03 \x03(\x01\x42\x02\x10\x01\"}\n\x0bStimRawData\x12\x11\n\tdata_rows\x18\x01 \x02(\r\x12\x11\n\tdata_cols\x18\x02 \x02(\r\x12\x10\n\x04\x64\x61ta\x18\x03 \x03(\x02\x42\x02\x10\x01\x12\x11\n\tmask_rows\x18\x04 \x02(\r\x12\x11\n\tmask_cols\x18\x05 \x02(\r\x12\x10\n\x04mask\x18\x06 \x03(\x02\x42\x02\x10\x01')
)
_sym_db.RegisterFileDescriptor(DESCRIPTOR)




_SOLVERPARAM = _descriptor.Descriptor(
  name='SolverParam',
  full_name='gsbn.SolverParam',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='gen_param', full_name='gsbn.SolverParam.gen_param', index=0,
      number=1, type=11, cpp_type=10, label=2,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='net_param', full_name='gsbn.SolverParam.net_param', index=1,
      number=2, type=11, cpp_type=10, label=2,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='rec_param', full_name='gsbn.SolverParam.rec_param', index=2,
      number=3, type=11, cpp_type=10, label=2,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=20,
  serialized_end=138,
)


_GENPARAM = _descriptor.Descriptor(
  name='GenParam',
  full_name='gsbn.GenParam',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='stim_file', full_name='gsbn.GenParam.stim_file', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='dt', full_name='gsbn.GenParam.dt', index=1,
      number=2, type=2, cpp_type=6, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='mode_param', full_name='gsbn.GenParam.mode_param', index=2,
      number=3, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=140,
  serialized_end=218,
)


_MODEPARAM = _descriptor.Descriptor(
  name='ModeParam',
  full_name='gsbn.ModeParam',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='begin_time', full_name='gsbn.ModeParam.begin_time', index=0,
      number=1, type=2, cpp_type=6, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='end_time', full_name='gsbn.ModeParam.end_time', index=1,
      number=2, type=2, cpp_type=6, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='prn', full_name='gsbn.ModeParam.prn', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='gain_mask', full_name='gsbn.ModeParam.gain_mask', index=3,
      number=4, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='plasticity', full_name='gsbn.ModeParam.plasticity', index=4,
      number=5, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='stim_index', full_name='gsbn.ModeParam.stim_index', index=5,
      number=6, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=221,
  serialized_end=354,
)


_NETPARAM = _descriptor.Descriptor(
  name='NetParam',
  full_name='gsbn.NetParam',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='pop_param', full_name='gsbn.NetParam.pop_param', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='proj_param', full_name='gsbn.NetParam.proj_param', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='procedure', full_name='gsbn.NetParam.procedure', index=2,
      number=3, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=356,
  serialized_end=457,
)


_POPPARAM = _descriptor.Descriptor(
  name='PopParam',
  full_name='gsbn.PopParam',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='pop_num', full_name='gsbn.PopParam.pop_num', index=0,
      number=1, type=13, cpp_type=3, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='hcu_num', full_name='gsbn.PopParam.hcu_num', index=1,
      number=2, type=13, cpp_type=3, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='mcu_num', full_name='gsbn.PopParam.mcu_num', index=2,
      number=3, type=13, cpp_type=3, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='slot_num', full_name='gsbn.PopParam.slot_num', index=3,
      number=4, type=13, cpp_type=3, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='fanout_num', full_name='gsbn.PopParam.fanout_num', index=4,
      number=5, type=13, cpp_type=3, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='taum', full_name='gsbn.PopParam.taum', index=5,
      number=6, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=0.01,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='wtagain', full_name='gsbn.PopParam.wtagain', index=6,
      number=7, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=4,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='maxfq', full_name='gsbn.PopParam.maxfq', index=7,
      number=8, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=100,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='igain', full_name='gsbn.PopParam.igain', index=8,
      number=9, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='wgain', full_name='gsbn.PopParam.wgain', index=9,
      number=10, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='lgbias', full_name='gsbn.PopParam.lgbias', index=10,
      number=11, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='snoise', full_name='gsbn.PopParam.snoise', index=11,
      number=12, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=460,
  serialized_end=693,
)


_PROJPARAM = _descriptor.Descriptor(
  name='ProjParam',
  full_name='gsbn.ProjParam',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='src_pop', full_name='gsbn.ProjParam.src_pop', index=0,
      number=1, type=13, cpp_type=3, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='dest_pop', full_name='gsbn.ProjParam.dest_pop', index=1,
      number=2, type=13, cpp_type=3, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='tauzi', full_name='gsbn.ProjParam.tauzi', index=2,
      number=3, type=2, cpp_type=6, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='tauzj', full_name='gsbn.ProjParam.tauzj', index=3,
      number=4, type=2, cpp_type=6, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='taue', full_name='gsbn.ProjParam.taue', index=4,
      number=5, type=2, cpp_type=6, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='taup', full_name='gsbn.ProjParam.taup', index=5,
      number=6, type=2, cpp_type=6, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='maxfq', full_name='gsbn.ProjParam.maxfq', index=6,
      number=7, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=100,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='bgain', full_name='gsbn.ProjParam.bgain', index=7,
      number=8, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='wgain', full_name='gsbn.ProjParam.wgain', index=8,
      number=9, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='pi0', full_name='gsbn.ProjParam.pi0', index=9,
      number=10, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=696,
  serialized_end=872,
)


_RECPARAM = _descriptor.Descriptor(
  name='RecParam',
  full_name='gsbn.RecParam',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='directory', full_name='gsbn.RecParam.directory', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=True, default_value=_b(".").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='enable', full_name='gsbn.RecParam.enable', index=1,
      number=2, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=True,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='offset', full_name='gsbn.RecParam.offset', index=2,
      number=3, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='snapshot_period', full_name='gsbn.RecParam.snapshot_period', index=3,
      number=4, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='spike_period', full_name='gsbn.RecParam.spike_period', index=4,
      number=5, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=874,
  serialized_end=1000,
)


_SOLVERSTATE = _descriptor.Descriptor(
  name='SolverState',
  full_name='gsbn.SolverState',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='timestamp', full_name='gsbn.SolverState.timestamp', index=0,
      number=1, type=2, cpp_type=6, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='prn', full_name='gsbn.SolverState.prn', index=1,
      number=2, type=2, cpp_type=6, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='table_state', full_name='gsbn.SolverState.table_state', index=2,
      number=3, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='vector_state_i', full_name='gsbn.SolverState.vector_state_i', index=3,
      number=4, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='vector_state_f', full_name='gsbn.SolverState.vector_state_f', index=4,
      number=5, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='vector_state_d', full_name='gsbn.SolverState.vector_state_d', index=5,
      number=6, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1003,
  serialized_end=1219,
)


_TABLESTATE = _descriptor.Descriptor(
  name='TableState',
  full_name='gsbn.TableState',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='name', full_name='gsbn.TableState.name', index=0,
      number=1, type=9, cpp_type=9, label=2,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='desc', full_name='gsbn.TableState.desc', index=1,
      number=2, type=12, cpp_type=9, label=2,
      has_default_value=False, default_value=_b(""),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='data', full_name='gsbn.TableState.data', index=2,
      number=3, type=12, cpp_type=9, label=2,
      has_default_value=False, default_value=_b(""),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1221,
  serialized_end=1275,
)


_VECTORSTATEI = _descriptor.Descriptor(
  name='VectorStateI',
  full_name='gsbn.VectorStateI',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='name', full_name='gsbn.VectorStateI.name', index=0,
      number=1, type=9, cpp_type=9, label=2,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='ld', full_name='gsbn.VectorStateI.ld', index=1,
      number=2, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='data', full_name='gsbn.VectorStateI.data', index=2,
      number=3, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=_descriptor._ParseOptions(descriptor_pb2.FieldOptions(), _b('\020\001'))),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1277,
  serialized_end=1338,
)


_VECTORSTATEF = _descriptor.Descriptor(
  name='VectorStateF',
  full_name='gsbn.VectorStateF',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='name', full_name='gsbn.VectorStateF.name', index=0,
      number=1, type=9, cpp_type=9, label=2,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='ld', full_name='gsbn.VectorStateF.ld', index=1,
      number=2, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='data', full_name='gsbn.VectorStateF.data', index=2,
      number=3, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=_descriptor._ParseOptions(descriptor_pb2.FieldOptions(), _b('\020\001'))),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1340,
  serialized_end=1401,
)


_VECTORSTATED = _descriptor.Descriptor(
  name='VectorStateD',
  full_name='gsbn.VectorStateD',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='name', full_name='gsbn.VectorStateD.name', index=0,
      number=1, type=9, cpp_type=9, label=2,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='ld', full_name='gsbn.VectorStateD.ld', index=1,
      number=2, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='data', full_name='gsbn.VectorStateD.data', index=2,
      number=3, type=1, cpp_type=5, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=_descriptor._ParseOptions(descriptor_pb2.FieldOptions(), _b('\020\001'))),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1403,
  serialized_end=1464,
)


_STIMRAWDATA = _descriptor.Descriptor(
  name='StimRawData',
  full_name='gsbn.StimRawData',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='data_rows', full_name='gsbn.StimRawData.data_rows', index=0,
      number=1, type=13, cpp_type=3, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='data_cols', full_name='gsbn.StimRawData.data_cols', index=1,
      number=2, type=13, cpp_type=3, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='data', full_name='gsbn.StimRawData.data', index=2,
      number=3, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=_descriptor._ParseOptions(descriptor_pb2.FieldOptions(), _b('\020\001'))),
    _descriptor.FieldDescriptor(
      name='mask_rows', full_name='gsbn.StimRawData.mask_rows', index=3,
      number=4, type=13, cpp_type=3, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='mask_cols', full_name='gsbn.StimRawData.mask_cols', index=4,
      number=5, type=13, cpp_type=3, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='mask', full_name='gsbn.StimRawData.mask', index=5,
      number=6, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=_descriptor._ParseOptions(descriptor_pb2.FieldOptions(), _b('\020\001'))),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1466,
  serialized_end=1591,
)

_SOLVERPARAM.fields_by_name['gen_param'].message_type = _GENPARAM
_SOLVERPARAM.fields_by_name['net_param'].message_type = _NETPARAM
_SOLVERPARAM.fields_by_name['rec_param'].message_type = _RECPARAM
_GENPARAM.fields_by_name['mode_param'].message_type = _MODEPARAM
_NETPARAM.fields_by_name['pop_param'].message_type = _POPPARAM
_NETPARAM.fields_by_name['proj_param'].message_type = _PROJPARAM
_SOLVERSTATE.fields_by_name['table_state'].message_type = _TABLESTATE
_SOLVERSTATE.fields_by_name['vector_state_i'].message_type = _VECTORSTATEI
_SOLVERSTATE.fields_by_name['vector_state_f'].message_type = _VECTORSTATEF
_SOLVERSTATE.fields_by_name['vector_state_d'].message_type = _VECTORSTATED
DESCRIPTOR.message_types_by_name['SolverParam'] = _SOLVERPARAM
DESCRIPTOR.message_types_by_name['GenParam'] = _GENPARAM
DESCRIPTOR.message_types_by_name['ModeParam'] = _MODEPARAM
DESCRIPTOR.message_types_by_name['NetParam'] = _NETPARAM
DESCRIPTOR.message_types_by_name['PopParam'] = _POPPARAM
DESCRIPTOR.message_types_by_name['ProjParam'] = _PROJPARAM
DESCRIPTOR.message_types_by_name['RecParam'] = _RECPARAM
DESCRIPTOR.message_types_by_name['SolverState'] = _SOLVERSTATE
DESCRIPTOR.message_types_by_name['TableState'] = _TABLESTATE
DESCRIPTOR.message_types_by_name['VectorStateI'] = _VECTORSTATEI
DESCRIPTOR.message_types_by_name['VectorStateF'] = _VECTORSTATEF
DESCRIPTOR.message_types_by_name['VectorStateD'] = _VECTORSTATED
DESCRIPTOR.message_types_by_name['StimRawData'] = _STIMRAWDATA

SolverParam = _reflection.GeneratedProtocolMessageType('SolverParam', (_message.Message,), dict(
  DESCRIPTOR = _SOLVERPARAM,
  __module__ = 'gsbn_pb2'
  # @@protoc_insertion_point(class_scope:gsbn.SolverParam)
  ))
_sym_db.RegisterMessage(SolverParam)

GenParam = _reflection.GeneratedProtocolMessageType('GenParam', (_message.Message,), dict(
  DESCRIPTOR = _GENPARAM,
  __module__ = 'gsbn_pb2'
  # @@protoc_insertion_point(class_scope:gsbn.GenParam)
  ))
_sym_db.RegisterMessage(GenParam)

ModeParam = _reflection.GeneratedProtocolMessageType('ModeParam', (_message.Message,), dict(
  DESCRIPTOR = _MODEPARAM,
  __module__ = 'gsbn_pb2'
  # @@protoc_insertion_point(class_scope:gsbn.ModeParam)
  ))
_sym_db.RegisterMessage(ModeParam)

NetParam = _reflection.GeneratedProtocolMessageType('NetParam', (_message.Message,), dict(
  DESCRIPTOR = _NETPARAM,
  __module__ = 'gsbn_pb2'
  # @@protoc_insertion_point(class_scope:gsbn.NetParam)
  ))
_sym_db.RegisterMessage(NetParam)

PopParam = _reflection.GeneratedProtocolMessageType('PopParam', (_message.Message,), dict(
  DESCRIPTOR = _POPPARAM,
  __module__ = 'gsbn_pb2'
  # @@protoc_insertion_point(class_scope:gsbn.PopParam)
  ))
_sym_db.RegisterMessage(PopParam)

ProjParam = _reflection.GeneratedProtocolMessageType('ProjParam', (_message.Message,), dict(
  DESCRIPTOR = _PROJPARAM,
  __module__ = 'gsbn_pb2'
  # @@protoc_insertion_point(class_scope:gsbn.ProjParam)
  ))
_sym_db.RegisterMessage(ProjParam)

RecParam = _reflection.GeneratedProtocolMessageType('RecParam', (_message.Message,), dict(
  DESCRIPTOR = _RECPARAM,
  __module__ = 'gsbn_pb2'
  # @@protoc_insertion_point(class_scope:gsbn.RecParam)
  ))
_sym_db.RegisterMessage(RecParam)

SolverState = _reflection.GeneratedProtocolMessageType('SolverState', (_message.Message,), dict(
  DESCRIPTOR = _SOLVERSTATE,
  __module__ = 'gsbn_pb2'
  # @@protoc_insertion_point(class_scope:gsbn.SolverState)
  ))
_sym_db.RegisterMessage(SolverState)

TableState = _reflection.GeneratedProtocolMessageType('TableState', (_message.Message,), dict(
  DESCRIPTOR = _TABLESTATE,
  __module__ = 'gsbn_pb2'
  # @@protoc_insertion_point(class_scope:gsbn.TableState)
  ))
_sym_db.RegisterMessage(TableState)

VectorStateI = _reflection.GeneratedProtocolMessageType('VectorStateI', (_message.Message,), dict(
  DESCRIPTOR = _VECTORSTATEI,
  __module__ = 'gsbn_pb2'
  # @@protoc_insertion_point(class_scope:gsbn.VectorStateI)
  ))
_sym_db.RegisterMessage(VectorStateI)

VectorStateF = _reflection.GeneratedProtocolMessageType('VectorStateF', (_message.Message,), dict(
  DESCRIPTOR = _VECTORSTATEF,
  __module__ = 'gsbn_pb2'
  # @@protoc_insertion_point(class_scope:gsbn.VectorStateF)
  ))
_sym_db.RegisterMessage(VectorStateF)

VectorStateD = _reflection.GeneratedProtocolMessageType('VectorStateD', (_message.Message,), dict(
  DESCRIPTOR = _VECTORSTATED,
  __module__ = 'gsbn_pb2'
  # @@protoc_insertion_point(class_scope:gsbn.VectorStateD)
  ))
_sym_db.RegisterMessage(VectorStateD)

StimRawData = _reflection.GeneratedProtocolMessageType('StimRawData', (_message.Message,), dict(
  DESCRIPTOR = _STIMRAWDATA,
  __module__ = 'gsbn_pb2'
  # @@protoc_insertion_point(class_scope:gsbn.StimRawData)
  ))
_sym_db.RegisterMessage(StimRawData)


_VECTORSTATEI.fields_by_name['data'].has_options = True
_VECTORSTATEI.fields_by_name['data']._options = _descriptor._ParseOptions(descriptor_pb2.FieldOptions(), _b('\020\001'))
_VECTORSTATEF.fields_by_name['data'].has_options = True
_VECTORSTATEF.fields_by_name['data']._options = _descriptor._ParseOptions(descriptor_pb2.FieldOptions(), _b('\020\001'))
_VECTORSTATED.fields_by_name['data'].has_options = True
_VECTORSTATED.fields_by_name['data']._options = _descriptor._ParseOptions(descriptor_pb2.FieldOptions(), _b('\020\001'))
_STIMRAWDATA.fields_by_name['data'].has_options = True
_STIMRAWDATA.fields_by_name['data']._options = _descriptor._ParseOptions(descriptor_pb2.FieldOptions(), _b('\020\001'))
_STIMRAWDATA.fields_by_name['mask'].has_options = True
_STIMRAWDATA.fields_by_name['mask']._options = _descriptor._ParseOptions(descriptor_pb2.FieldOptions(), _b('\020\001'))
# @@protoc_insertion_point(module_scope)