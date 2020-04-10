from hashing_utils import plot_mAP, plot_precision_number, plot_precision_recall, plot_recall_number
import matplotlib.pyplot as plt
from parameter import *

def plot_graph(database_name, dims, lambda_T, poses, precision_dims, recall_dims, pre_dims, rec_dims, mAP_dims):
  n_dims = len(dims)
  colors = {}
  linestyles = {}
  markers = {}
  markes = ['o', 's', '^', 'p', '^', 'v', 'p', 'd', 'h', '2', '8', '6']

  fig = plt.figure(num=1, figsize=(21, 14), dpi=80)
  ticks = [i for i in range(n_dims)]
  fig.suptitle('db_name:{},lambda_T:{},lambda_U:{},lambda_V:{},lambda_Z:{}'.format(database_name, lambda_T,lambda_U, lambda_V, lambda_Z))
  methods_names = []

  # plot MAP 
  ax1 = fig.add_subplot(1, 4, 1)
  ax1.set_xlabel('the number of bits')
  ax1.set_ylabel('mAP')
  ax1.set_xticks(ticks)
  ax1.set_xticklabels(labels=dims)
  plt.plot(ticks, mAP_dims)
  # for key in methods:
  # plt.plot(ticks, mAP_res.get(key), label=key, color=colors[key], linestyle=linestyles[key], marker=markers[key])
  # workbook = xlrd.open_workbook('file.xlsx')
  # sheet = workbook.sheet_by_index(0)
  # for i in range(sheet.ncols):
  #     col = sheet.col_values(i)
  #     methods_names.append(col[0])
  #     plt.plot(ticks, col[1:], label=col[0], marker=markes[i])

  # num_methods = len(methods_names)
  #plot precision vs. the number of retrieved sample.
  choose_bits = 0
  ax2 = fig.add_subplot(1, 4, 2)
  ax2.set_xlabel('The number of retrieved samples')
  ax2.set_ylabel('Precision @ {} bits'.format(dims[choose_bits]))
  plt.plot(poses, pre_dims[choose_bits])
  # for key in methods:
  #     plt.plot(poses, pre_res.get(key)[choose_bits], color=colors[key], linestyle=linestyles[key], marker=markers[key])
  # sheet = workbook.sheet_by_index(2)
  # for i in range(num_methods):
  #     row = sheet.row_values(i * num_methods + choose_bits)
  #     plt.plot(poses, row, marker=markes[i])
  

  #plot recall vs. the number of retrieved sample.
  ax3 = fig.add_subplot(1, 4, 3)
  ax3.set_xlabel('The number of retrieved samples')
  ax3.set_ylabel('Recall @ {} bits'.format(dims[choose_bits]))
  plt.plot(poses, rec_dims[choose_bits])
  # for key in methods:
  #     plt.plot(poses, rec_res.get(key)[choose_bits], color=colors[key], linestyle=linestyles[key], marker=markers[key])
  # sheet = workbook.sheet_by_index(1)
  # for i in range(num_methods):
  #     row = sheet.row_values(i * num_methods + choose_bits)
  #     plt.plot(poses, row, marker=markes[i])

  
  #plot precision vs. recall , i is the selection of which bits.
  ax4 = fig.add_subplot(1, 4, 4)
  ax4.set_xlabel('Recall @ {} bits'.format(dims[choose_bits]))
  ax4.set_ylabel('Precision @ {} bits'.format(dims[choose_bits]))
  plt.plot(recall_dims[choose_bits], precision_dims[choose_bits])
  # for key in methods:
  #     plt.plot(recall_res.get(key)[choose_bits], precision_res.get(key)[choose_bits], color=colors[key], linestyle=linestyles[key], marker=markers[key])
  # sheet_recall = workbook.sheet_by_index(3)
  # sheet_precision = workbook.sheet_by_index(4)
  # for i in range(num_methods):
  #     row_recall = sheet_recall.row_values(i * num_methods + choose_bits)
  #     row_precision = sheet_precision.row_values(i * num_methods + choose_bits)
  #     row_recall = [i for i in row_recall if i is not '']
  #     row_precision = [i for i in row_precision if i is not '']
  #     plt.plot(row_recall, row_precision, marker=markes[i])

  # ax5 = plt.subplot(235)
  # ax5.set_xlabel('The number of retrieved radius')
  # ax5.set_ylabel('Precision @ {} bits'.format(dims[choose_bits]))
  # for key in methods:
  #     radius = [i for i in range(len(precision_res.get(key)[choose_bits]))]
  #     plt.plot(radius, precision_res.get(key)[choose_bits], color=colors[key], linestyle=linestyles[key], marker=markers[key])
  # sheet = workbook.sheet_by_index(4)
  # for i in range(num_methods):
  #     row = sheet.row_values(i * num_methods + choose_bits)
  #     row = [i for i in row if i is not '']
  #     radius = [i for i in range(len(row))]
  #     plt.plot(radius, row, marker=markes[i])

  fig.legend(loc='upper right')
  # plt.savefig('db_name-{}-k-{}-lambda_D-{}-kk-{}.png'.format(database_name, k, lambda_D, kk))
  plt.show()
