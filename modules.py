from .utils import *
import pdb
import math
from collections import namedtuple


class CharCnn(nn.Module):
	def __init__(
			self,
			charset_size,
			char_emb_dim,
			filter_wids,
			filter_nums,
			outp_size
	):
		super().__init__()
		self.char_emb_mat = nn.Embedding(charset_size, char_emb_dim, padding_idx=0)
		self.cnns = nn.ModuleList(
			[
				nn.Conv2d(
					in_channels=1,
					out_channels=filter_num,
					kernel_size=(filter_wid, char_emb_dim),
					padding=(filter_wid - 1, 0),
					bias=True
				)
				for filter_wid, filter_num in zip(filter_wids, filter_nums)
			]
		)
		self.tot_filter_num = sum(filter_nums)
		self.lin = nn.Linear(self.tot_filter_num, outp_size)
	
	# inp: char_idxes
	def forward(self, char_idxes, dropout_prob=.5):
		# [batch_num, word_len, char_emb_dim]
		char_embs = self.char_emb_mat(char_idxes)
		# [batch_num, 1, word_len, char_emb_dim]
		char_embs = char_embs.unsqueeze(dim=1)
		# [batch_num, filter_num, outp_len] * len(filter_wids)
		cnn_outps = [
			# [batch_num, filter_num, outp_len]
			F.selu(
				# [batch_num, filter_num, outp_len, 1]
				cnn(char_embs)
			).squeeze(dim=3)
			for cnn in self.cnns
		]
		dropout = nn.Dropout(dropout_prob)
		# [batch_num, filter_num] * len(filter_wids)
		pool_outps = [
			# [batch_size, filter_num]
			F.max_pool1d(
				dropout(cnn_outp),
				cnn_outp.size()[2]
			).squeeze(dim=2)
			for cnn_outp in cnn_outps
		]
		# [batch_size, outp_size]
		outp = self.lin(
			# [batch_size, tot_filter_num]
			dropout(
				torch.cat(pool_outps, dim=1)
			)
		)
		
		return F.tanh(outp)


class Gate(nn.Module):
	def __init__(
			self,
			hid_size
	):
		super().__init__()
		
		self.hid_size = hid_size
		self.wg = self.wq = nn.Parameter(
			torch.randn(self.hid_size, self.hid_size).cuda()
		)
	
	def forward(
			self,
			# [batch_size, hid_size]
			inp
	):
		# [batch_size, hid_size]
		g = softmax(
			torch.matmul(
				# [batch_size, hid_size]
				inp,
				# [hid_size, hid_size]
				self.wg
			)
		)
		# [batch_size, hid_size]
		return torch.mul(inp, g)


class EmbLyr(nn.Module):
	def __init__(
			self,
			word_set_size,
			word_emb_dim,
			pad_word_idx,
			tag_set_size,
			tag_emb_dim,
			pad_tag_idx,
			pretrained_word_emb_mat=None
	):
		super().__init__()
		
		self.word_emb_mat = pretrained_word_emb_mat or nn.Embedding(
			num_embeddings=word_set_size,
			embedding_dim=word_emb_dim,
			padding_idx=pad_word_idx
		)
		
		self.tag_emb_mat = nn.Embedding(
			num_embeddings=tag_set_size,
			embedding_dim=tag_emb_dim,
			padding_idx=pad_tag_idx
		)
	
	def forward(self, word_idxes, tag_idxes):
		word_embs = self.word_emb_mat(word_idxes)
		tag_embs = self.tag_emb_mat(tag_idxes)
		# [text_len, batch_size, word_emb_dim + tag_emb_dim]
		return torch.cat((word_embs, tag_embs), dim=-1).cuda()


class AttLyr(nn.Module):
	Ret = namedtuple("Ret", "scores wgts outp")
	
	def __init__(
			self,
			hid_size,
			ctx_hid_size,
			qry_hid_size
	):
		super().__init__()
		
		self.hid_size = hid_size
		self.ctx_hid_size = ctx_hid_size
		self.qry_hid_size = qry_hid_size
		self.batch_size = 1
		self.wc = nn.Parameter(
			torch.randn(self.ctx_hid_size, self.hid_size).cuda()
		)
		self.wq = nn.Parameter(
			torch.randn(self.qry_hid_size, self.hid_size).cuda()
		)
		self.v = nn.Parameter(
			torch.randn(self.hid_size, 1).cuda()
		)
		self.debugging = False
		self.uses_score_mask = False
	
	def debug(self):
		self.debugging = True
		
		return self
	
	def undebug(self):
		self.debugging = False
		
		return self
	
	def set_ctx(
			self,
			# [ctx_len, batch_size | 1, ctx_hid_size]
			ctx,
			unpadded_lens=None
	):
		# if self.debugging:
		# 	print(f"ctx.size() = {ctx.size()}")
		
		self.ctx_len, self.batch_size, _ = ctx.size()
		# [batch_size | 1, ctx_len, ctx_hid_size]
		self.transp_ctx = torch.transpose(ctx, 0, 1)
		# [batch_size | 1, 1, ctx_len, hid_size]
		self.precalc_prod = torch.matmul(
			# [batch_size | 1, ctx_len, ctx_hid_size]
			self.transp_ctx,
			# [ctx_hid_size, hid_size]
			self.wc
		).view(-1, 1, self.ctx_len, self.hid_size)
		
		if unpadded_lens:
			assert len(unpadded_lens) == self.batch_size
			
			self.uses_score_mask = True
			# [batch_size, 1, ctx_len]
			self.score_mask = Variable(
				torch.cuda.FloatTensor(
					[
						[
							0. if pos < unpadded_lens[idx_in_batch] else -math.inf
							for pos in range(self.ctx_len)
						]
						for idx_in_batch in range(self.batch_size)
					]
				)
			).view(self.batch_size, 1, self.ctx_len)
		
		return self
	
	def forward(
			self,
			# [1 | batch_size, (qry_num, )qry_hid_size]
			qries,
	):
		qry_batch_size = qries.size()[0]
		
		if qry_batch_size == 1 or self.batch_size == 1:
			self.batch_size = max(self.batch_size, qry_batch_size)
		else:
			assert self.batch_size == qry_batch_size
		
		qry_num = 1
		has_multi_qries = len(qries.size()) == 3
		# [1 | batch_size, (qry_num, )qry_hid_size]
		if has_multi_qries:
			qry_num = qries.size()[1]
		
		# if self.debugging:
		# print(f"qries.size() = {qries.size()}")
		# print(f"self.precalc_prod.size() = {self.precalc_prod.size()}")
		
		# [batch_size, qry_num, ctx_len]
		scores = torch.matmul(
			# [batch_size, qry_num, ctx_len, hid_size]
			F.tanh(
				# [batch_size, qry_num, ctx_len, hid_size]
				torch.add(
					# [batch_size | 1, 1, ctx_len, hid_size]
					self.precalc_prod,
					# [1 | batch_size, qry_num, 1, hid_size]
					torch.matmul(
						# [1 | batch_size, (qry_num, )qry_hid_size]
						qries,
						# [qry_hid_size, hid_size]
						self.wq
					).view(-1, qry_num, 1, self.hid_size)
				)
			),
			# [hid_size, 1]
			self.v
		).view(self.batch_size, qry_num, self.ctx_len)
		
		if self.debugging:
			print(f"wq = {self.wq}")
			print(f"scores = {scores}")
		
		if self.uses_score_mask:
			# [batch_size, qry_num, ctx_len]
			scores = torch.add(
				# [batch_size, qry_num, ctx_len]
				scores,
				# [batch_size, 1, ctx_len]
				self.score_mask
			)
		
		# [batch_size * qry_num, ctx_len]
		scores = scores.view(-1, self.ctx_len)
		
		# [batch_size * qry_num, ctx_len]
		wgts = softmax(scores)
		# [batch_size, (qry_num, )ctx_hid_size]
		outp = torch.matmul(
			# [batch_size, qry_num, 1, ctx_len]
			wgts.view(-1, qry_num, 1, self.ctx_len),
			# [batch_size, 1, ctx_len, ctx_hid_size]
			self.transp_ctx.view(-1, 1, self.ctx_len, self.ctx_hid_size)
		)
		
		if self.debugging:
			print(f"wgts.size() = {wgts.size()}")
			print(f"outp.size() = {outp.size()}")
		
		if has_multi_qries:
			scores = scores.view(self.batch_size, qry_num, -1)
			wgts = wgts.view(self.batch_size, qry_num, -1)
			outp = outp.view(self.batch_size, qry_num, -1)
		else:
			scores = scores.view(self.batch_size, -1)
			wgts = wgts.view(self.batch_size, -1)
			outp = outp.view(self.batch_size, -1)
		
		return AttLyr.Ret(scores, wgts, outp)


# attention might be over multiple contexts
class AttBiLstm(nn.Module):
	def __init__(
			self,
			inp_size,
			hid_size,
			ctx_hid_sizes
			# feeds_init_state=False
	):
		super().__init__()
		
		self.inp_size = inp_size
		self.hid_size = hid_size
		self.ctx_hid_sizes = ctx_hid_sizes
		# self.feeds_init_state = feeds_init_state
		self.att_lyrs = nn.ModuleList(
			[
				AttLyr(
					hid_size=self.hid_size,
					ctx_hid_size=ctx_hid_size,
					qry_hid_size=(self.hid_size // 2)
				)
				for ctx_hid_size in self.ctx_hid_sizes
			]
		)
		self.inp_gate = Gate(hid_size=(self.inp_size + sum(self.ctx_hid_sizes)))
		self.fwd_cell = nn.LSTMCell(
			input_size=(self.inp_size + sum(self.ctx_hid_sizes)),
			hidden_size=(self.hid_size // 2),
			bias=True
		)
		self.bwd_cell = nn.LSTMCell(
			input_size=(self.inp_size + sum(self.ctx_hid_sizes)),
			hidden_size=(self.hid_size // 2),
			bias=True
		)
	
	def forward(
			self,
			# list of [ctx_len, batch_size, inp_size]
			ctxs,
			# [inp_len, batch_size, inp_size]
			inps,
			init_fwd_state,
			init_bwd_state
	):
		for att_lyr, ctx in zip(self.att_lyrs, ctxs):
			att_lyr.set_ctx(ctx)
		
		inp_len, batch_size, _ = inps.size()
		fwd_hid_state, fwd_cell_state = init_fwd_state
		bwd_hid_state, bwd_cell_state = init_bwd_state
		fwd_outps, bwd_outps = [], []
		
		for t in range(inp_len):
			# [batch_size, hid_size / 2], [batch_size, hid_size / 2]
			fwd_hid_state, fwd_cell_state = self.fwd_cell(
				# [batch_size, inp_size + sum(ctx_hid_sizes)]
				self.inp_gate(
					# [batch_size, inp_size + sum(ctx_hid_sizes)]
					torch.cat(
						# [batch_size, inp_size], [batch_size, ctx_hid_size], ...
						[inps[t]] + [
							att_lyr(
								# [batch_size, hid_size / 2]
								qries=fwd_hid_state,
							).outp
							for att_lyr in self.att_lyrs
						],
						dim=-1
					)
				),
				(fwd_hid_state, fwd_cell_state)
			)
			
			fwd_outps.append(fwd_hid_state)
			
			bwd_hid_state, bwd_cell_state = self.bwd_cell(
				self.inp_gate(
					torch.cat(
						[inps[-t - 1]] + [
							att_lyr(
								# [batch_size, hid_size / 2]
								qries=bwd_hid_state,
							).outp
							for att_lyr in self.att_lyrs
						],
						dim=-1
					)
				),
				(bwd_hid_state, bwd_cell_state)
			)
			bwd_outps.append(bwd_hid_state)
		
		# [batch_size, hid_size / 2], [batch_size, hid_size / 2]
		last_fwd_state = (fwd_hid_state, fwd_cell_state)
		last_bwd_state = (bwd_hid_state, bwd_cell_state)
		last_state = (last_fwd_state, last_bwd_state)
		# [inp_len, batch_size, hid_size]
		return torch.stack(
			# inp_len * [batch_size, hid_size]
			[
				# [batch_size, hid_size]
				torch.cat(
					# [batch_size, hid_size / 2], [batch_size, hid_size / 2]
					(fwd_outp, bwd_outp),
					dim=-1
				)
				for fwd_outp, bwd_outp in zip(fwd_outps, reversed(bwd_outps))
			]
		), last_state


# attention might be over multiple contexts
class AttBiGru(nn.Module):
	def __init__(
			self,
			inp_size,
			hid_size,
			ctx_hid_sizes
			# feeds_init_state=False
	):
		super().__init__()
		
		self.inp_size = inp_size
		self.hid_size = hid_size
		self.ctx_hid_sizes = ctx_hid_sizes
		# self.feeds_init_state = feeds_init_state
		self.att_lyrs = nn.ModuleList(
			[
				AttLyr(
					hid_size=self.hid_size,
					ctx_hid_size=ctx_hid_size,
					qry_hid_size=(self.hid_size // 2)
				)
				for ctx_hid_size in self.ctx_hid_sizes
			]
		)
		self.inp_gate = Gate(hid_size=(self.inp_size + sum(self.ctx_hid_sizes)))
		self.fwd_cell = nn.GRUCell(
			input_size=(self.inp_size + sum(self.ctx_hid_sizes)),
			hidden_size=(self.hid_size // 2),
			bias=True
		)
		self.bwd_cell = nn.GRUCell(
			input_size=(self.inp_size + sum(self.ctx_hid_sizes)),
			hidden_size=(self.hid_size // 2),
			bias=True
		)
	
	def forward(
			self,
			# list of [ctx_len, batch_size, inp_size]
			ctxs,
			# [inp_len, batch_size, inp_size]
			inps,
			init_fwd_state,
			init_bwd_state
	):
		for att_lyr, ctx in zip(self.att_lyrs, ctxs):
			att_lyr.set_ctx(ctx)
		
		inp_len, batch_size, _ = inps.size()
		fwd_state = init_fwd_state
		bwd_state = init_bwd_state
		fwd_outps, bwd_outps = [], []
		
		for t in range(inp_len):
			# [batch_size, hid_size / 2]
			fwd_state = self.fwd_cell(
				self.inp_gate(
					# [batch_size, inp_size + sum(ctx_hid_sizes)]
					torch.cat(
						# [batch_size, inp_size], [batch_size, ctx_hid_size], ...
						[inps[t]] + [
							att_lyr(
								# [batch_size, hid_size / 2]
								qries=fwd_state,
							).outp
							for att_lyr in self.att_lyrs
						],
						dim=-1
					)
				),
				fwd_state
			)
			
			fwd_outps.append(fwd_state)
			
			bwd_state = self.bwd_cell(
				self.inp_gate(
					torch.cat(
						[inps[-t - 1]] + [
							att_lyr(
								# [batch_size, hid_size / 2]
								qries=bwd_state,
							).outp
							for att_lyr in self.att_lyrs
						],
						dim=-1
					)
				),
				bwd_state
			)
			bwd_outps.append(bwd_state)
		
		# [batch_size, hid_size / 2], [batch_size, hid_size / 2]
		last_fwd_state = fwd_state
		last_bwd_state = bwd_state
		last_state = (last_fwd_state, last_bwd_state)
		# [inp_len, batch_size, hid_size]
		return torch.stack(
			# inp_len * [batch_size, hid_size]
			[
				# [batch_size, hid_size]
				torch.cat(
					# [batch_size, hid_size / 2], [batch_size, hid_size / 2]
					(fwd_outp, bwd_outp),
					dim=-1
				)
				for fwd_outp, bwd_outp in zip(fwd_outps, reversed(bwd_outps))
			]
		), last_state


class TextLstmEnc(nn.Module):
	def __init__(
			self,
			inp_size,
			hid_size,
			uses_self_att_lyr=True
	):
		super().__init__()
		
		self.inp_size = inp_size
		self.hid_size = hid_size
		self.enc_lstm = nn.LSTM(
			input_size=self.inp_size,
			hidden_size=(self.hid_size // 2),
			bidirectional=True
		)
		
		self.uses_self_att_lyr = uses_self_att_lyr
		
		self.self_att_lstm = AttBiLstm(
			inp_size=self.hid_size,
			hid_size=self.hid_size,
			ctx_hid_sizes=(self.hid_size,)
		) if self.uses_self_att_lyr else None
	
	def forward(
			self,
			# [text_len, batch_size=1, inp_size]
			inps
	):
		# print("in text encoder")
		
		enc_outps, (last_enc_hid_state, last_enc_cell_state) = self.enc_lstm(
			inps, None
		)
		
		# print("after encoding lstm")
		
		# [batch_size, hid_size / 2], [batch_size, hid_size / 2]
		last_enc_fwd_hid_state, last_enc_bwd_hid_state = torch.unbind(last_enc_hid_state, dim=0)
		last_enc_fwd_cell_state, last_enc_bwd_cell_state = torch.unbind(last_enc_cell_state, dim=0)
		
		if self.uses_self_att_lyr:
			# [text_len, batch_size, hid_size],
			# (
			# 	([batch_size, hid_size / 2]: fwd_hid_state, [batch_size, hid_size / 2]: fwd_cell_state),
			# 	([batch_size, hid_size / 2]: bwd_hid_state, [batch_size, hid_size / 2]: bwd_cell_state)
			# ): last_state
			return self.self_att_lstm(
				ctxs=(enc_outps,),
				inps=enc_outps,
				# [batch_size, hid_size / 2]
				init_fwd_state=(last_enc_bwd_hid_state, last_enc_bwd_cell_state),
				# [batch_size, hid_size / 2]
				init_bwd_state=(last_enc_fwd_hid_state, last_enc_fwd_cell_state)
			)
		else:
			# [text_len, batch_size, hid_size],
			# (
			# 	([batch_size, hid_size / 2]: fwd_hid_state, [batch_size, hid_size / 2]: fwd_cell_state),
			# 	([batch_size, hid_size / 2]: bwd_hid_state, [batch_size, hid_size / 2]: bwd_cell_state)
			# ): last_state
			return enc_outps, (
				(last_enc_fwd_hid_state, last_enc_fwd_cell_state),
				(last_enc_bwd_hid_state, last_enc_bwd_cell_state)
			)


class TextGruEnc(nn.Module):
	def __init__(
			self,
			inp_size,
			hid_size,
			uses_self_att_lyr=True
	):
		super().__init__()
		
		self.inp_size = inp_size
		self.hid_size = hid_size
		self.enc_gru = nn.GRU(
			input_size=self.inp_size,
			hidden_size=(self.hid_size // 2),
			bidirectional=True
		)
		
		self.uses_self_att_lyr = uses_self_att_lyr
		
		self.self_att_gru = AttBiGru(
			inp_size=self.hid_size,
			hid_size=self.hid_size,
			ctx_hid_sizes=(self.hid_size,)
		) if self.uses_self_att_lyr else None
	
	def forward(
			self,
			# [text_len, batch_size, inp_size]
			inps
	):
		enc_outps, last_enc_state = self.enc_gru(
			inps, None
		)
		# [batch_size, hid_size / 2], [batch_size, hid_size / 2]
		last_enc_fwd_state, last_enc_bwd_state = torch.unbind(last_enc_state, dim=0)
		if self.uses_self_att_lyr:
			# [text_len, batch_size, hid_size],
			return self.self_att_gru(
				ctxs=(enc_outps,),
				inps=enc_outps,
				# [batch_size, hid_size / 2]
				init_fwd_state=last_enc_bwd_state,
				# [batch_size, hid_size / 2]
				init_bwd_state=last_enc_fwd_state
			)
		else:
			# [text_len, batch_size, hid_size], ([batch_size, hid_size / 2], [batch_size, hid_size / 2])
			return enc_outps, (last_enc_fwd_state, last_enc_bwd_state)

