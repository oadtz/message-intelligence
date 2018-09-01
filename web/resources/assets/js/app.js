
/**
 * First we will load all of this project's JavaScript dependencies which
 * includes Vue and other libraries. It is a great starting point when
 * building robust, powerful web applications using Vue and Laravel.
 */

import './bootstrap';
import Vue from 'vue';

var app = new Vue({
    el: '#app',
    data: {
        question: '',
        answers: null
    },
    watch: {
        question: function (newVal, oldVal) {
            this.debouncedGetAnswers();
        }
    },
    created: function () {
        this.debouncedGetAnswers = _.debounce(this.getAnswers, 200);
    },
    methods: {
        getAnswers: function () {
            var v = this;
            v.answers = null;

            if (_.trim(v.question).length >= 3) {
                axios.get('/api/flights', {
                    params: {
                        text: v.question,
                        prob: 0.99
                    }
                })
                .then(function (response) {
                    v.answers = response.data;
                });
            }
        },
        setQuestion: function (q) {
            this.question = q;
        },
        addAnswer: function () {
            var v = this;
            
            if (_.trim(v.question)) {
                axios.post('/api/flights', {
                    text: v.question
                })
                .then(function (response) {
                    alert (v.question + ' was added to be trained');
                })
            }
        }
    }
});
